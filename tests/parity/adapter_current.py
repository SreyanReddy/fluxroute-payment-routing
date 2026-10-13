"""
Adapter for the CURRENT serving code: serving/feature_storing.py.

The parity test is a referee that never changes. This file is the only part
that knows how the current serving code stores data in Redis. When
feature_storing.py is rewritten, a new adapter replaces this one and the
referee (run_parity.py) stays exactly the same.

For one transaction it:
    1. empties an in-memory Redis (fakeredis: no Redis server needed)
    2. writes the history into it, using the current code's key names
    3. freezes the serving code's clock at the transaction's minute
    4. asks the serving code for features, exactly as /route would
"""

import sys
from pathlib import Path

import fakeredis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "serving"))

import feature_storing as fs  # noqa: E402

NAME = "current serving code (serving/feature_storing.py)"

# Point the serving code at an in-memory Redis instead of a real server.
_redis = fakeredis.FakeRedis(decode_responses=True)
fs._redis_client = _redis

# The replay feeds production-scale traffic (the same stream training used),
# so the traffic meter needs no scaling here.
fs.TRAFFIC_SCALE = 1.0


def serving_features(history, txn) -> dict:
    _redis.flushall()
    pipe = _redis.pipeline()

    # Route health comes from the aggregate stream, as in training.
    route_scope = f"route:{history.route_id}"
    for minute, rm in history.route_minutes.items():
        pipe.set(f"{route_scope}:req:{minute}", rm.attempts)
        pipe.set(f"{route_scope}:resolved:{minute}", rm.attempts)  # replayed history is fully resolved
        pipe.set(f"{route_scope}:succ:{minute}", rm.successes)
        pipe.set(f"{route_scope}:lat:{minute}", rm.latency_sum)

    # System traffic also comes from the aggregate stream. The meter adds up
    # per-second buckets over the last 60 seconds; only that sum matters, so
    # each minute's total goes into the minute's first second.
    for minute, requests in history.system_requests.items():
        pipe.set(f"fr:ingress:sec:{minute * 60}", requests)

    # Bank, network and bank x route come from the sampled stream. Bank and
    # network also record which minutes had traffic ("active" minutes), the
    # same way /outcome does.
    _put_counts(pipe, f"bank:{history.bank}", history.bank_minutes, track_active=True)
    _put_counts(pipe, f"network:{history.network}", history.network_minutes, track_active=True)
    _put_counts(pipe, f"bankroute:{history.bank}:{history.route_id}", history.bank_route_minutes)

    # The current code has no separate place for the sampled route stream, so
    # history.route_sampled_minutes has nowhere to go. route_failure_rate_1m is
    # therefore computed from the aggregate route counters instead.

    if history.last_flagged_minute is not None:
        pipe.set(f"route:{history.route_id}:last_flagged_minute", history.last_flagged_minute)

    if history.last_bank_route_success_minute is not None:
        pipe.set(
            f"bankroute:{history.bank}:{history.route_id}:last_success_minute",
            history.last_bank_route_success_minute,
        )

    if history.bank_amount_count > 0:
        pipe.set(f"bank:{history.bank}:amount_sum", history.bank_amount_sum)
        pipe.set(f"bank:{history.bank}:amount_count", history.bank_amount_count)

    pipe.execute()

    # Freeze the clock at the start of the transaction's minute. The serving
    # code's current_minute() is derived from current_second(), so both now
    # answer with the transaction's moment.
    now_second = history.minute * 60
    fs.current_second = lambda: now_second

    # Read the traffic meter once, exactly as /route does.
    system_tps, _ = fs.read_system_tps(now_second)

    return fs.build_route_features(
        route_id=txn.route_id,
        bank=txn.bank,
        network=txn.network,
        amount=float(txn.amount),
        payment_method=txn.payment_method,
        merchant=txn.merchant,
        device=txn.device,
        risk=txn.risk,
        system_tps=system_tps,
    )


def _put_counts(pipe, scope: str, minutes: dict, track_active: bool = False) -> None:
    for minute, counts in minutes.items():
        pipe.set(f"{scope}:req:{minute}", counts.attempts)
        pipe.set(f"{scope}:resolved:{minute}", counts.attempts)
        pipe.set(f"{scope}:succ:{minute}", counts.successes)
        if track_active:
            pipe.zadd(f"{scope}:active", {minute: minute})
