"""
Referee for the write-equivalence test: a frozen copy of the Redis WRITE
paths of serving/feature_storing.py as they were before experiment C2
(one round trip per command), at commit "Load test: expC_pipeline".

Function bodies are copied verbatim. Only the plumbing differs:
get_redis_client() and current_second() point at whatever the test runner
installs in _redis and _clock, instead of a real server and the real clock.

Never "fix" this file to match new serving code. It is the reference the
new code is checked against, the same way tests/parity/ checks the reads.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "serving"))

from feature_storing import (  # noqa: E402
    ACTIVE_RETENTION_MINUTES,
    BUCKET_TTL_SECONDS,
    FLAG_DOWN_THRESHOLD,
    INGRESS_TTL_SECONDS,
    INGRESS_WINDOW_SECONDS,
    SYSTEM_TPS_MAX,
    SYSTEM_TPS_MIN,
    WINDOW_5M,
)

TRAFFIC_SCALE = 1.0  # the runner sets this to match the code under test

_redis = None   # installed by the runner
_clock = None   # installed by the runner: a zero-argument callable returning a Unix second


def get_redis_client():
    return _redis


def current_second() -> int:
    return _clock()


def current_minute() -> int:
    return current_second() // 60


# ---- copied verbatim from feature_storing.py (pre-C2) ----

def record_ingress(second: int) -> None:
    """Count one arriving request in its second's bucket."""
    r = get_redis_client()
    key = f"fr:ingress:sec:{second}"
    r.incr(key)
    r.expire(key, INGRESS_TTL_SECONDS)


def read_system_tps(now_second: int) -> tuple[float, str | None]:
    r = get_redis_client()
    keys = [f"fr:ingress:sec:{now_second - k}" for k in range(1, INGRESS_WINDOW_SECONDS + 1)]
    requests = sum(float(v) for v in r.mget(keys) if v is not None)
    observed = requests / INGRESS_WINDOW_SECONDS * TRAFFIC_SCALE

    if observed < SYSTEM_TPS_MIN:
        return SYSTEM_TPS_MIN, "low"
    if observed > SYSTEM_TPS_MAX:
        return SYSTEM_TPS_MAX, "high"
    return observed, None


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
    r = get_redis_client()
    now = current_minute()
    keys = [_bucket_key(scope, metric, now - offset) for offset in range(1, window_minutes + 1)]
    values = r.mget(keys)
    return sum(float(v) for v in values if v is not None)


def get_rolling_rate(scope: str, window_minutes: int) -> tuple[float | None, float]:
    resolved = sum_window(scope, "resolved", window_minutes)
    successes = sum_window(scope, "succ", window_minutes)
    if resolved <= 0:
        return None, 0.0
    return successes / resolved, resolved


def mark_route_flagged_down(route_id: str) -> None:
    get_redis_client().set(f"route:{route_id}:last_flagged_minute", current_minute())


def mark_bank_route_success(bank: str, route_id: str) -> None:
    get_redis_client().set(f"bankroute:{bank}:{route_id}:last_success_minute", current_minute())


def record_bank_amount(bank: str, amount: float) -> None:
    r = get_redis_client()
    r.incrbyfloat(f"bank:{bank}:amount_sum", amount)
    r.incr(f"bank:{bank}:amount_count")


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
