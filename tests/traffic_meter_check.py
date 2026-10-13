"""
Checks the live traffic meter against the worked example:

    TRAFFIC_SCALE = 800
    Locust load    ->  model sees           R4 utilization
    5 req/s        ->  clamped up to min    (quieter than anything in training)
    20 req/s       ->  16,000               0.032
    200 req/s      ->  160,000              0.320
    350 req/s      ->  280,000              0.560
    500 req/s      ->  clamped down to max  (busier than anything in training)

Also checks that the current second is never counted, so a request does not
see itself.

Run:
    ~/.venvs/fluxroute/bin/python tests/traffic_meter_check.py
"""

import math
import sys
from pathlib import Path

import fakeredis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "serving"))

import feature_storing as fs  # noqa: E402

fs._redis_client = fakeredis.FakeRedis(decode_responses=True)
fs.TRAFFIC_SCALE = 800.0

NOW = 1_800_000_000  # any fixed second
R4 = fs.ROUTE_STATIC_CONFIG["R4"]


def run_at(rate_per_second: int, extra_in_current_second: int = 0):
    """Fill the last 60 completed seconds at a steady rate, then read the meter."""
    r = fs._redis_client
    r.flushall()
    for k in range(1, 61):
        r.set(f"fr:ingress:sec:{NOW - k}", rate_per_second)
    if extra_in_current_second:
        r.set(f"fr:ingress:sec:{NOW}", extra_in_current_second)
    return fs.read_system_tps(NOW)


def utilization_r4(system_tps: float) -> float:
    return system_tps * R4["routing_share"] / R4["base_capacity_tps"]


cases = [
    (5, fs.SYSTEM_TPS_MIN, "low"),
    (20, 16_000.0, None),
    (200, 160_000.0, None),
    (350, 280_000.0, None),
    (500, fs.SYSTEM_TPS_MAX, "high"),
]

print(f"TRAFFIC_SCALE = {fs.TRAFFIC_SCALE:g}   training range = "
      f"{fs.SYSTEM_TPS_MIN:,.0f} .. {fs.SYSTEM_TPS_MAX:,.0f}\n")
print(f"{'load':>10}{'system_tps':>14}{'clamp':>8}{'R4 util':>10}")

failures = 0
for rate, expected_tps, expected_clamp in cases:
    tps, clamp = run_at(rate)
    ok = math.isclose(tps, expected_tps) and clamp == expected_clamp
    failures += not ok
    print(f"{rate:>7} /s{tps:>14,.0f}{str(clamp or '-'):>8}{utilization_r4(tps):>10.3f}"
          f"{'' if ok else '   <-- WRONG'}")

tps_alone, _ = run_at(200)
tps_with_current, _ = run_at(200, extra_in_current_second=1_000_000)
self_excluded = math.isclose(tps_alone, tps_with_current)
failures += not self_excluded
print(f"\ncurrent second ignored (a request never counts itself): "
      f"{'yes' if self_excluded else 'NO -- WRONG'}")

print("\nALL CHECKS PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
