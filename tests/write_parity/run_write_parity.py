"""
Write-equivalence test: the counterpart of tests/parity/ for the WRITE paths.

tests/parity/ checks that serving reads build the same features training did,
but its adapter seeds Redis itself and never calls the write functions. This
test covers them. It replays one seeded stream of routing decisions and
outcomes through two implementations, each against its own in-memory Redis:

    referee:  legacy_writes.py        (frozen pre-C2 code, one command per round trip)
    current:  serving/feature_storing (pipelined writes)

and requires the two Redis states to be identical -- every key, type, value,
and TTL -- at the end of every simulated minute, and every value the write
paths return (traffic meter reading, popped pending decision) to match.

The stream is built to reach every branch: successes and failures, a stretch
where one route fails often enough to be flagged down, outcomes that arrive
seconds late, duplicate outcome reports, unknown transaction ids, and
decisions whose outcome never arrives.

Run:
    python tests/write_parity/run_write_parity.py
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import fakeredis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "serving"))
sys.path.insert(0, str(HERE))

import feature_storing as fs  # noqa: E402
import legacy_writes as legacy  # noqa: E402

SEED = 7
START_SECOND = 1_800_000_000 - (1_800_000_000 % 60)  # a minute boundary
DURATION_SECONDS = 9 * 60
MAX_DECISIONS_PER_SECOND = 6

BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "BOB"]
METHODS = [("CARD", "Visa"), ("CARD", "Mastercard"), ("CARD", "Rupay"), ("UPI", "UPI")]

# Minutes (relative to START) in which R3 mostly fails, so its 5-minute
# success rate drops below FLAG_DOWN_THRESHOLD and the flag-down write runs.
R3_OUTAGE_MINUTES = range(2, 6)

# ---- one clock, two Redis servers ----

clock = {"second": START_SECOND}


def now_second() -> int:
    return clock["second"]


legacy._redis = fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
legacy._clock = now_second
legacy.TRAFFIC_SCALE = 800.0

fs._redis_client = fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
fs.current_second = now_second
fs.TRAFFIC_SCALE = 800.0

# ---- comparison ----

mismatches: list[str] = []


def check(label: str, legacy_value, current_value) -> None:
    if legacy_value != current_value:
        mismatches.append(f"{label}: legacy={legacy_value!r} current={current_value!r}")


def snapshot(r) -> dict:
    state = {}
    for key in r.scan_iter("*"):
        kind = r.type(key)
        if kind == "string":
            value = r.get(key)
        elif kind == "hash":
            value = r.hgetall(key)
        elif kind == "zset":
            value = r.zrange(key, 0, -1, withscores=True)
        else:
            value = f"<unexpected type {kind}>"
        state[key] = (kind, value, r.ttl(key))
    return state


def compare_states(label: str) -> None:
    legacy_state = snapshot(legacy._redis)
    current_state = snapshot(fs._redis_client)

    for key in sorted(legacy_state.keys() - current_state.keys()):
        mismatches.append(f"{label}: key only in legacy: {key}")
    for key in sorted(current_state.keys() - legacy_state.keys()):
        mismatches.append(f"{label}: key only in current: {key}")

    for key in sorted(legacy_state.keys() & current_state.keys()):
        l_kind, l_value, l_ttl = legacy_state[key]
        c_kind, c_value, c_ttl = current_state[key]
        if (l_kind, l_value) != (c_kind, c_value):
            mismatches.append(f"{label}: {key}: legacy={l_kind} {l_value!r} current={c_kind} {c_value!r}")
        # TTLs count down in wall-clock time, and the two writes happen
        # microseconds apart, so allow one second of drift. "No expiry" (-1)
        # must match exactly.
        no_expiry_differs = (l_ttl < 0 or c_ttl < 0) and l_ttl != c_ttl
        if no_expiry_differs or abs(l_ttl - c_ttl) > 1:
            mismatches.append(f"{label}: {key}: TTL legacy={l_ttl} current={c_ttl}")


def same_float(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


# ---- the stream ----

rng = random.Random(SEED)
scheduled_outcomes: dict[int, list[str]] = defaultdict(list)
stats = defaultdict(int)


def make_decision(index: int) -> None:
    second = clock["second"]
    payment_method, network = rng.choice(METHODS)
    route_id = rng.choice(fs.get_eligible_routes(payment_method, network))
    bank = rng.choices(BANKS, weights=[30, 18, 22, 12, 10, 8])[0]
    amount = round(max(10.0, min(rng.lognormvariate(5.5, 1.35), 50_000.0)), 2)
    probability = rng.random()
    transaction_id = f"txn-{index}"

    legacy_tps = legacy.read_system_tps(second)
    legacy.record_ingress(second)
    legacy.record_routing_decision(route_id, bank, network, amount)
    legacy.record_pending_route(transaction_id, route_id, bank, network, probability)

    current_tps = fs.meter_request(second)
    fs.record_routing_decision(transaction_id, route_id, bank, network, amount, probability)

    if not (same_float(legacy_tps[0], current_tps[0]) and legacy_tps[1] == current_tps[1]):
        mismatches.append(f"meter at {second}: legacy={legacy_tps!r} current={current_tps!r}")

    stats["decisions"] += 1
    if rng.random() < 0.03:
        stats["never_resolved"] += 1  # pending record stays until its TTL
        return
    scheduled_outcomes[second + rng.randint(0, 20)].append(transaction_id)
    if rng.random() < 0.02:
        scheduled_outcomes[second + rng.randint(21, 40)].append(transaction_id)  # duplicate report


def report_outcome(transaction_id: str) -> None:
    legacy_pending = legacy.pop_pending_route(transaction_id)
    current_pending = fs.pop_pending_route(transaction_id)
    check(f"pop {transaction_id}", legacy_pending, current_pending)

    if legacy_pending is None:
        stats["outcomes_without_pending"] += 1
        return

    route_id = legacy_pending["route_id"]
    minute_offset = clock["second"] // 60 - START_SECOND // 60
    if route_id == "R3" and minute_offset in R3_OUTAGE_MINUTES:
        success = rng.random() < 0.2
    else:
        success = rng.random() < 0.97
    latency_ms = round(max(1.0, rng.gauss(120, 20)), 2)

    flagged_key = f"route:{route_id}:last_flagged_minute"
    flagged_before = legacy._redis.get(flagged_key)

    for implementation in (legacy, fs):
        implementation.record_transaction_outcome(
            route_id, legacy_pending["bank"], legacy_pending["network"], success, latency_ms
        )

    stats["outcomes"] += 1
    stats["failures"] += not success
    if not success and legacy._redis.get(flagged_key) != flagged_before:
        stats["flag_down_writes"] += 1


decision_index = 0
for second in range(START_SECOND, START_SECOND + DURATION_SECONDS):
    clock["second"] = second

    for _ in range(rng.randint(0, MAX_DECISIONS_PER_SECOND)):
        make_decision(decision_index)
        decision_index += 1

    for transaction_id in scheduled_outcomes.pop(second, []):
        report_outcome(transaction_id)

    if rng.random() < 0.05:
        report_outcome(f"unknown-{second}")

    if (second + 1) % 60 == 0:
        compare_states(f"minute {(second - START_SECOND) // 60}")

compare_states("end")

# ---- report ----

print("Replayed stream:")
for name in (
    "decisions", "outcomes", "failures", "flag_down_writes",
    "outcomes_without_pending", "never_resolved",
):
    print(f"  {name:<26}{stats[name]:>7}")

# A stream that never reached a branch would pass without testing it.
required = ("failures", "flag_down_writes", "outcomes_without_pending", "never_resolved")
unexercised = [name for name in required if stats[name] == 0]
if unexercised:
    print(f"\nSTREAM TOO WEAK: never exercised {', '.join(unexercised)}")
    sys.exit(2)

if mismatches:
    print(f"\n{len(mismatches)} MISMATCH(ES). First 20:")
    for line in mismatches[:20]:
        print(f"  {line}")
    sys.exit(1)

print(f"\nWRITE PARITY: PASS -- {len(snapshot(fs._redis_client))} keys identical "
      f"(type, value, TTL) at every minute boundary")
