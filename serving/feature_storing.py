"""
Real-time feature computation backed by Redis, mirroring the offline
feature engineering in dataset-generator-routing/feature_extraction3.py
and ml/lightgbm_tuned.py::engineer_features(). Any change to the offline
formulas MUST be mirrored here, or the model silently sees train/serve
skew -- exactly the class of bug the ML phase spent real effort catching.
"""

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
BUCKET_TTL_SECONDS = 1800  # 30 min, comfortably past the 15m window

# ============================================================
# STATIC ROUTE CONFIG -- mirrors dataset-generator-routing/routes.py.
# Keep in sync manually for now; replace with a real config service later.
# ============================================================

ROUTE_STATIC_CONFIG = {
    "R1": {
        "base_success_rate": 0.986,
        "base_latency_ms": 145,
        "cost_percent": 2.10,
        "base_capacity_tps": 120_000,
        "supported_payment_methods": {"CARD", "UPI"},
        "supported_networks": {"Visa", "Mastercard", "UPI"},
    },
    "R2": {
        "base_success_rate": 0.981,
        "base_latency_ms": 85,
        "cost_percent": 1.45,
        "base_capacity_tps": 150_000,
        "supported_payment_methods": {"CARD", "UPI"},
        "supported_networks": {"Visa", "Rupay", "UPI"},
    },
    "R3": {
        "base_success_rate": 0.979,
        "base_latency_ms": 110,
        "cost_percent": 1.10,
        "base_capacity_tps": 100_000,
        "supported_payment_methods": {"CARD", "UPI"},
        "supported_networks": {"Visa", "Mastercard", "Rupay", "UPI"},
    },
    "R4": {
        "base_success_rate": 0.983,
        "base_latency_ms": 175,
        "cost_percent": 1.75,
        "base_capacity_tps": 75_000,
        "supported_payment_methods": {"CARD", "UPI"},
        "supported_networks": {"Visa", "Mastercard", "Rupay", "UPI"},
    },
}

ALL_ROUTE_IDS = list(ROUTE_STATIC_CONFIG.keys())


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


import os

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

def current_minute() -> int:
    return int(time.time() // 60)


# ============================================================
# PRIMITIVE 1: minute-bucketed counters
# ============================================================

def _bucket_key(scope: str, metric: str, minute: int) -> str:
    return f"{scope}:{metric}:{minute}"


def record_request(scope: str) -> None:
    r = get_redis_client()
    key = _bucket_key(scope, "req", current_minute())
    r.incr(key)
    r.expire(key, BUCKET_TTL_SECONDS)


def record_outcome(scope: str, success: bool, latency_ms: float | None = None) -> None:
    r = get_redis_client()
    minute = current_minute()

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
    """Returns (success_rate_or_None, request_count). None = no traffic yet."""
    requests = sum_window(scope, "req", window_minutes)
    successes = sum_window(scope, "succ", window_minutes)
    if requests <= 0:
        return None, 0.0
    return successes / requests, requests


def get_rolling_avg_latency(scope: str, window_minutes: int) -> float | None:
    requests = sum_window(scope, "req", window_minutes)
    latency_sum = sum_window(scope, "lat", window_minutes)
    if requests <= 0:
        return None
    return latency_sum / requests


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
) -> dict:
    """
    Returns every NUMERIC_FEATURES / CATEGORICAL_FEATURES key the model
    expects, for ONE candidate route. Call once per eligible route per
    incoming transaction, during scoring -- before a routing decision
    is made. No Redis writes happen here.
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

    route_request_rate_tps = sum_window(f"route:{route_id}", "req", 1) / 60.0
    route_utilization = route_request_rate_tps / config["base_capacity_tps"]
    system_request_rate_tps = sum_window("global", "req", 1) / 60.0

    bank_failure_rate_5m, _ = get_rolling_rate(f"bank:{bank}", WINDOW_5M)
    if bank_failure_rate_5m is not None:
        bank_failure_rate_5m = 1.0 - bank_failure_rate_5m

    network_failure_rate_5m, _ = get_rolling_rate(f"network:{network}", WINDOW_5M)
    if network_failure_rate_5m is not None:
        network_failure_rate_5m = 1.0 - network_failure_rate_5m

    bank_route_requests_15m = sum_window(f"bankroute:{bank}:{route_id}", "req", WINDOW_15M)
    bank_route_successes_15m = sum_window(f"bankroute:{bank}:{route_id}", "succ", WINDOW_15M)
    prior = config["base_success_rate"]
    bank_route_success_rate_15m = (
        bank_route_successes_15m + BANK_ROUTE_SHRINK_K * prior
    ) / (bank_route_requests_15m + BANK_ROUTE_SHRINK_K)

    time_since_last_failure = minutes_since_route_flagged(route_id)
    time_since_last_bank_route_success = minutes_since_bank_route_success(bank, route_id)

    bank_avg_amount = get_bank_avg_amount(bank)
    amount_to_bank_avg_ratio = amount / bank_avg_amount if bank_avg_amount else 1.0

    # Cold-start fallbacks: no traffic history yet -> use static baselines.
    success_rate_5m = success_rate_5m if success_rate_5m is not None else config["base_success_rate"]
    success_rate_15m = success_rate_15m if success_rate_15m is not None else config["base_success_rate"]
    avg_latency_5m = avg_latency_5m if avg_latency_5m is not None else config["base_latency_ms"]
    failure_rate_1m = failure_rate_1m if failure_rate_1m is not None else (1.0 - config["base_success_rate"])
    bank_failure_rate_5m = bank_failure_rate_5m if bank_failure_rate_5m is not None else 0.0
    network_failure_rate_5m = network_failure_rate_5m if network_failure_rate_5m is not None else 0.0

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
        "system_tps": system_request_rate_tps,
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
    record_request(f"route:{route_id}")
    record_request(f"bank:{bank}")
    record_request(f"network:{network}")
    record_request(f"bankroute:{bank}:{route_id}")
    record_request("global")
    record_bank_amount(bank, amount)


def record_transaction_outcome(
    route_id: str, bank: str, network: str, success: bool, latency_ms: float,
) -> None:
    """Call exactly once, when the real outcome is known (callback/webhook)."""
    record_outcome(f"route:{route_id}", success, latency_ms)
    record_outcome(f"bank:{bank}", success)
    record_outcome(f"network:{network}", success)
    record_outcome(f"bankroute:{bank}:{route_id}", success)

    if success:
        mark_bank_route_success(bank, route_id)
    else:
        success_rate_5m, _ = get_rolling_rate(f"route:{route_id}", WINDOW_5M)
        if success_rate_5m is not None and success_rate_5m < FLAG_DOWN_THRESHOLD:
            mark_route_flagged_down(route_id)


PENDING_TTL_SECONDS = 3600


def record_pending_route(transaction_id: str, route_id: str, bank: str, network: str) -> None:
    r = get_redis_client()
    key = f"pending:{transaction_id}"
    r.hset(key, mapping={"route_id": route_id, "bank": bank, "network": network})
    r.expire(key, PENDING_TTL_SECONDS)


def pop_pending_route(transaction_id: str) -> dict | None:
    r = get_redis_client()
    key = f"pending:{transaction_id}"
    pending = r.hgetall(key)
    if not pending:
        return None
    r.delete(key)
    return pending