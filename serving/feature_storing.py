"""
Real-time feature computation backed by Redis, mirroring the offline
feature engineering in dataset-generator-routing/feature_extraction3.py
and ml/lightgbm_tuned.py::engineer_features(). Any change to the offline
formulas MUST be mirrored here, or the model silently sees train/serve
skew -- exactly the class of bug the ML phase spent real effort catching.
"""

import json
import os
import time

import redis

# ============================================================
# CONSTANTS -- must match the offline pipeline exactly
# ============================================================

WINDOW_5M = 5
WINDOW_15M = 15
FLAG_DOWN_THRESHOLD = 0.70
BANK_ROUTE_SHRINK_K = 10
NO_FAILURE_SENTINEL = 9999

# Bank and network failure rates look back over the last 5 minutes that had
# any traffic, which can stretch far back at quiet times, so counters are
# kept for a day rather than just past the 15-minute window.
BUCKET_TTL_SECONDS = 24 * 60 * 60
ACTIVE_RECORDS = 5
ACTIVE_RETENTION_MINUTES = 24 * 60

# ============================================================
# STATIC ROUTE CONFIG -- loaded from models/serving_constants_v2.json,
# which ml/export_serving_constants.py generates from the training data.
# Never type these numbers in by hand: the model learned the values that
# training measured, not the simulator's nominal settings.
# ============================================================

CONSTANTS_PATH = os.environ.get(
    "SERVING_CONSTANTS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "serving_constants_v2.json"),
)

with open(CONSTANTS_PATH) as _f:
    _CONSTANTS = json.load(_f)

ROUTE_STATIC_CONFIG = {
    route_id: {
        **values,
        "supported_payment_methods": set(values["supported_payment_methods"]),
        "supported_networks": set(values["supported_networks"]),
    }
    for route_id, values in _CONSTANTS["routes"].items()
}

ALL_ROUTE_IDS = list(ROUTE_STATIC_CONFIG.keys())

# ============================================================
# LIVE TRAFFIC METER -- the model's "how busy is the system" dial.
#
# Training saw production-scale traffic: system_tps from ~8k (3am) to ~308k
# (weekend peak). A live service can only count the requests that reach it,
# so it multiplies its own rate by TRAFFIC_SCALE -- "each request I see
# stands for this many in the whole payment system" -- and clamps the result
# to the range training saw.
#
# TRAFFIC_SCALE is a setting, not a constant. For a load test, choose
#     training median / planned test rate     (162,978 / 200 is about 800)
# so the planned load lands in the middle of the range.
# ============================================================

TRAFFIC_SCALE = float(os.environ.get("TRAFFIC_SCALE", "1.0"))
SYSTEM_TPS_MIN = _CONSTANTS["system_tps_training"]["min"]
SYSTEM_TPS_MAX = _CONSTANTS["system_tps_training"]["max"]
INGRESS_WINDOW_SECONDS = 60
INGRESS_TTL_SECONDS = 120


def get_eligible_routes(payment_method: str, network: str) -> list[str]:
    return [
        route_id
        for route_id, config in ROUTE_STATIC_CONFIG.items()
        if payment_method in config["supported_payment_methods"]
        and network in config["supported_networks"]
    ]


# ============================================================
# REDIS CLIENT
# ============================================================

_redis_client = None

def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True,
        )
    return _redis_client

def current_second() -> int:
    return int(time.time())


def current_minute() -> int:
    return current_second() // 60


def record_ingress(second: int) -> None:
    """Count one arriving request in its second's bucket."""
    r = get_redis_client()
    key = f"fr:ingress:sec:{second}"
    r.incr(key)
    r.expire(key, INGRESS_TTL_SECONDS)


def read_system_tps(now_second: int) -> tuple[float, str | None]:
    """
    Requests per second over the last 60 completed seconds, scaled up to
    production size and clamped to the range training saw.

    The current second is excluded, so a request never counts itself.
    Read it once per request: every candidate route gets the same value.

    Returns (system_tps, clamp) where clamp is "low", "high", or None.
    """
    r = get_redis_client()
    keys = [f"fr:ingress:sec:{now_second - k}" for k in range(1, INGRESS_WINDOW_SECONDS + 1)]
    requests = sum(float(v) for v in r.mget(keys) if v is not None)
    observed = requests / INGRESS_WINDOW_SECONDS * TRAFFIC_SCALE

    if observed < SYSTEM_TPS_MIN:
        return SYSTEM_TPS_MIN, "low"
    if observed > SYSTEM_TPS_MAX:
        return SYSTEM_TPS_MAX, "high"
    return observed, None


# ============================================================
# PRIMITIVE 1: minute-bucketed counters
# ============================================================

def _bucket_key(scope: str, metric: str, minute: int) -> str:
    return f"{scope}:{metric}:{minute}"

def record_outcome(
    scope: str,
    success: bool,
    latency_ms: float | None = None,
    track_active: bool = False,
) -> None:
    r = get_redis_client()
    minute = current_minute()

    if track_active:
        active_key = f"{scope}:active"
        r.zadd(active_key, {minute: minute})
        r.zremrangebyscore(active_key, "-inf", minute - ACTIVE_RETENTION_MINUTES)

    resolved_key = _bucket_key(scope, "resolved", minute)
    r.incr(resolved_key)
    r.expire(resolved_key, BUCKET_TTL_SECONDS)

    if success:
        key = _bucket_key(scope, "succ", minute)
        r.incr(key)
        r.expire(key, BUCKET_TTL_SECONDS)

    if latency_ms is not None:
        key = _bucket_key(scope, "lat", minute)
        r.incrbyfloat(key, latency_ms)
        r.expire(key, BUCKET_TTL_SECONDS)


def sum_window(scope: str, metric: str, window_minutes: int) -> float:
    """
    Sum the trailing `window_minutes` buckets, EXCLUDING the current
    minute -- matches prior_rolling_sum()'s convention offline.
    """
    r = get_redis_client()
    now = current_minute()
    keys = [_bucket_key(scope, metric, now - offset) for offset in range(1, window_minutes + 1)]
    values = r.mget(keys)
    return sum(float(v) for v in values if v is not None)


def get_rolling_rate(scope: str, window_minutes: int) -> tuple[float | None, float]:
    """
    Returns (success_rate_or_None, resolved_count).

    The denominator is RESOLVED outcomes, not requests. A request is counted
    when the routing decision is made, but its success is only counted when
    /outcome arrives -- dividing by requests would count every in-flight
    transaction as a failure and bias every route's rate downward.
    """
    resolved = sum_window(scope, "resolved", window_minutes)
    successes = sum_window(scope, "succ", window_minutes)
    if resolved <= 0:
        return None, 0.0
    return successes / resolved, resolved


def failure_rate_over_last_active_minutes(scope: str, before_minute: int) -> float:
    """
    Failure rate over the last ACTIVE_RECORDS minutes, before `before_minute`,
    in which this scope had any resolved transaction.

    This is how training computed bank_failure_rate_5m and
    network_failure_rate_5m: engineer_features() rolls over minutes that had
    transactions, not clock minutes. At quiet times the last 5 active minutes
    can reach far back, while the last 5 clock minutes may be nearly empty.
    Returns NaN when there is no earlier active minute, as training does.
    """
    r = get_redis_client()
    minutes = r.zrevrangebyscore(
        f"{scope}:active", before_minute - 1, "-inf", start=0, num=ACTIVE_RECORDS
    )
    if not minutes:
        return float("nan")

    resolved = r.mget([_bucket_key(scope, "resolved", int(m)) for m in minutes])
    successes = r.mget([_bucket_key(scope, "succ", int(m)) for m in minutes])
    total = sum(float(v) for v in resolved if v is not None)
    if total <= 0:
        return float("nan")
    succeeded = sum(float(v) for v in successes if v is not None)
    return (total - succeeded) / total


def get_rolling_avg_latency(scope: str, window_minutes: int) -> float | None:
    resolved = sum_window(scope, "resolved", window_minutes)
    latency_sum = sum_window(scope, "lat", window_minutes)
    if resolved <= 0:
        return None
    return latency_sum / resolved


# ============================================================
# PRIMITIVE 2: last-event-minute tracking
# ============================================================

def mark_route_flagged_down(route_id: str) -> None:
    get_redis_client().set(f"route:{route_id}:last_flagged_minute", current_minute())


def minutes_since_route_flagged(route_id: str) -> float:
    value = get_redis_client().get(f"route:{route_id}:last_flagged_minute")
    if value is None:
        return float(NO_FAILURE_SENTINEL)
    return float(current_minute() - int(value))


def mark_bank_route_success(bank: str, route_id: str) -> None:
    get_redis_client().set(f"bankroute:{bank}:{route_id}:last_success_minute", current_minute())


def minutes_since_bank_route_success(bank: str, route_id: str) -> float:
    value = get_redis_client().get(f"bankroute:{bank}:{route_id}:last_success_minute")
    if value is None:
        return float(NO_FAILURE_SENTINEL)
    return float(current_minute() - int(value))


# ============================================================
# PRIMITIVE 3: cumulative running average (bank amount baseline)
# ============================================================

def get_bank_avg_amount(bank: str) -> float | None:
    r = get_redis_client()
    total = r.get(f"bank:{bank}:amount_sum")
    count = r.get(f"bank:{bank}:amount_count")
    if total is None or count is None or float(count) <= 0:
        return None
    return float(total) / float(count)


def record_bank_amount(bank: str, amount: float) -> None:
    r = get_redis_client()
    r.incrbyfloat(f"bank:{bank}:amount_sum", amount)
    r.incr(f"bank:{bank}:amount_count")


# ============================================================
# SCORING: called once per CANDIDATE route -- read-only, no side effects
# ============================================================

def build_route_features(
    route_id: str, bank: str, network: str, amount: float,
    payment_method: str, merchant: str, device: str, risk: str,
    system_tps: float,
) -> dict:
    """
    Returns every NUMERIC_FEATURES / CATEGORICAL_FEATURES key the model
    expects, for ONE candidate route. Call once per eligible route per
    incoming transaction, during scoring -- before a routing decision
    is made. No Redis writes happen here.

    system_tps is the traffic-meter reading from read_system_tps(), taken
    once per request and shared by every candidate route.
    """
    now = current_minute()
    config = ROUTE_STATIC_CONFIG[route_id]

    day_in_week = (now // (24 * 60)) % 7
    hour_of_day = (now % (24 * 60)) // 60

    success_rate_5m, _ = get_rolling_rate(f"route:{route_id}", WINDOW_5M)
    success_rate_15m, _ = get_rolling_rate(f"route:{route_id}", WINDOW_15M)
    avg_latency_5m = get_rolling_avg_latency(f"route:{route_id}", WINDOW_5M)
    failure_rate_1m, _ = get_rolling_rate(f"route:{route_id}", 1)
    if failure_rate_1m is not None:
        failure_rate_1m = 1.0 - failure_rate_1m

    # Training split traffic by fixed shares (R1 always 35%, R4 always 15%),
    # so utilization meant "how stressed would this route be under the
    # standard split, given how busy the system is". Live, the model decides
    # where traffic goes; using each route's actual traffic would make the
    # feature describe the model's own choices, which training never showed.
    route_utilization = system_tps * config["routing_share"] / config["base_capacity_tps"]

    bank_failure_rate_5m = failure_rate_over_last_active_minutes(f"bank:{bank}", now)
    network_failure_rate_5m = failure_rate_over_last_active_minutes(f"network:{network}", now)

    bank_route_resolved_15m = sum_window(f"bankroute:{bank}:{route_id}", "resolved", WINDOW_15M)
    bank_route_successes_15m = sum_window(f"bankroute:{bank}:{route_id}", "succ", WINDOW_15M)
    prior = config["base_success_rate"]
    bank_route_success_rate_15m = (
        bank_route_successes_15m + BANK_ROUTE_SHRINK_K * prior
    ) / (bank_route_resolved_15m + BANK_ROUTE_SHRINK_K)

    time_since_last_failure = minutes_since_route_flagged(route_id)
    time_since_last_bank_route_success = minutes_since_bank_route_success(bank, route_id)

    bank_avg_amount = get_bank_avg_amount(bank)
    amount_to_bank_avg_ratio = amount / bank_avg_amount if bank_avg_amount else 1.0

    # Cold-start fallbacks: no traffic history yet -> use static baselines.
    success_rate_5m = success_rate_5m if success_rate_5m is not None else config["base_success_rate"]
    success_rate_15m = success_rate_15m if success_rate_15m is not None else config["base_success_rate"]
    avg_latency_5m = avg_latency_5m if avg_latency_5m is not None else config["base_latency_ms"]
    failure_rate_1m = failure_rate_1m if failure_rate_1m is not None else (1.0 - config["base_success_rate"])

    route_flagged_down = float(success_rate_5m < FLAG_DOWN_THRESHOLD)

    route_failure_rate_5m = 1.0 - success_rate_5m
    route_failure_rate_15m = 1.0 - success_rate_15m
    route_success_drop_5m = config["base_success_rate"] - success_rate_5m
    route_success_drop_15m = config["base_success_rate"] - success_rate_15m
    route_latency_ratio = avg_latency_5m / (config["base_latency_ms"] + 1e-6)
    bank_route_gap = bank_route_success_rate_15m - success_rate_15m
    route_stress = route_utilization * route_failure_rate_15m

    return {
        "day_in_week": day_in_week,
        "hour_of_day": hour_of_day,
        "amount": amount,
        "route_base_success_rate": config["base_success_rate"],
        "route_base_latency_ms": config["base_latency_ms"],
        "route_cost_percent": config["cost_percent"],
        "route_success_rate_5m": success_rate_5m,
        "route_success_rate_15m": success_rate_15m,
        "route_avg_latency_5m": avg_latency_5m,
        "time_since_last_failure": time_since_last_failure,
        "route_flagged_down": route_flagged_down,
        "route_utilization": route_utilization,
        "system_tps": system_tps,
        "bank_route_success_rate_15m": bank_route_success_rate_15m,
        "route_failure_rate_5m": route_failure_rate_5m,
        "route_failure_rate_15m": route_failure_rate_15m,
        "route_success_drop_5m": route_success_drop_5m,
        "route_success_drop_15m": route_success_drop_15m,
        "route_latency_ratio": route_latency_ratio,
        "bank_route_gap": bank_route_gap,
        "route_stress": route_stress,
        "bank_failure_rate_5m": bank_failure_rate_5m,
        "network_failure_rate_5m": network_failure_rate_5m,
        "route_failure_rate_1m": failure_rate_1m,
        "time_since_last_bank_route_success": time_since_last_bank_route_success,
        "amount_to_bank_avg_ratio": amount_to_bank_avg_ratio,
        "bank": bank,
        "network": network,
        "payment_method": payment_method,
        "merchant": merchant,
        "device": device,
        "risk": risk,
        "route_id": route_id,
    }


# ============================================================
# RECORDING: called ONCE per transaction, never per candidate
# ============================================================

def record_routing_decision(route_id: str, bank: str, network: str, amount: float) -> None:
    """Call exactly once, right after argmax picks the winning route."""
    record_bank_amount(bank, amount)


def record_transaction_outcome(
    route_id: str, bank: str, network: str, success: bool, latency_ms: float,
) -> None:
    """Call exactly once, when the real outcome is known (callback/webhook)."""
    record_outcome(f"route:{route_id}", success, latency_ms)
    record_outcome(f"bank:{bank}", success, track_active=True)
    record_outcome(f"network:{network}", success, track_active=True)
    record_outcome(f"bankroute:{bank}:{route_id}", success)

    if success:
        mark_bank_route_success(bank, route_id)
    else:
        success_rate_5m, _ = get_rolling_rate(f"route:{route_id}", WINDOW_5M)
        if success_rate_5m is not None and success_rate_5m < FLAG_DOWN_THRESHOLD:
            mark_route_flagged_down(route_id)


PENDING_TTL_SECONDS = 3600


def record_pending_route(
    transaction_id: str, route_id: str, bank: str, network: str, predicted_probability: float
) -> None:
    # predicted_probability travels with the pending decision so /outcome can
    # compare what the model expected with what actually happened (calibration
    # monitoring). It is one more field in the same HSET, not an extra call.
    r = get_redis_client()
    key = f"pending:{transaction_id}"
    r.hset(
        key,
        mapping={
            "route_id": route_id,
            "bank": bank,
            "network": network,
            "predicted_probability": predicted_probability,
        },
    )
    r.expire(key, PENDING_TTL_SECONDS)


def pop_pending_route(transaction_id: str) -> dict | None:
    r = get_redis_client()
    key = f"pending:{transaction_id}"
    pending = r.hgetall(key)
    if not pending:
        return None
    r.delete(key)
    return pending