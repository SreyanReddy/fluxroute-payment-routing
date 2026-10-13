"""
FEATURE PARITY TEST

The question it answers:
    Given the same history, does the serving code compute the same 26 numbers
    that training computed?

How it works:
    1. Training's answers: run the real training feature code
       (ml/lightgbm_tuned.py :: load_data) and keep the test-week rows.
    2. Pick transactions to check: ordinary moments, moments during a route
       incident, and quiet night-time moments.
    3. For each one, rebuild the history that existed just before it
       (history.py), hand it to the serving code (adapter_current.py), and
       collect the serving code's 26 numbers.
    4. Compare value by value and print a report.

Each feature ends up as one of:
    PASS        matches training on every transaction checked
    FAIL        differs somewhere -- a bug
    KNOWN GAP   cannot match exactly, for a documented reason (KNOWN_GAPS
                below). It still has to stay within its limit; if its typical
                difference grows past the limit it becomes GAP TOO BIG, which
                fails the test like a normal FAIL.

What it does NOT check:
    - Whether live traffic looks like training traffic (for example the
      system_tps scale). It feeds serving the same data training used, so it
      tests formulas, not data sources.
    - The write path (how /outcome updates Redis). That is a separate test.

Run:
    ~/.venvs/fluxroute/bin/python tests/parity/run_parity.py

Exit code 0 means no FAIL and no GAP TOO BIG; 1 otherwise.
"""

import math
import os
import sys
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

import adapter_current as adapter
from history import PROJECT_ROOT, HistoryBuilder

sys.path.insert(0, str(PROJECT_ROOT / "ml"))
import lightgbm_tuned  # noqa: E402  (training's own feature code)

MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_tuned_v2.joblib"
CACHE_FILE = PROJECT_ROOT / "tests" / "parity" / ".cache" / "training_answers_test_week.pkl"

ORDINARY_ROWS = 400
INCIDENT_ROWS = 50
QUIET_ROWS = 50
QUIET_HOURS = (2, 3, 4)
SEED = 7

# Two numbers count as equal if they agree to about 6 significant digits.
# Only floating-point rounding should ever be this small.
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-9


@dataclass
class KnownGap:
    reason: str
    limit: float     # largest allowed median difference
    relative: bool   # True: limit is a fraction of the training value


KNOWN_GAPS = {
    "system_tps": KnownGap(
        "Training knew this minute's traffic (the simulator decides it at the start "
        "of the minute). A live service only knows traffic that has already arrived, "
        "so it uses the previous minute.",
        limit=0.10,
        relative=True,
    ),
    "route_utilization": KnownGap(
        "Built from system_tps, so it inherits the same previous-minute gap.",
        limit=0.10,
        relative=True,
    ),
    "route_stress": KnownGap(
        "Built from route_utilization, so it inherits the same previous-minute gap.",
        limit=0.10,
        relative=True,
    ),
    "route_failure_rate_1m": KnownGap(
        "Training used the sampled transactions; serving uses its own outcome stream. "
        "Measured: feeding it from the production-scale stream does not reduce model "
        "quality (ROC-AUC 0.75082 -> 0.75087).",
        limit=0.05,
        relative=False,
    ),
}


def load_training_answers() -> pd.DataFrame:
    """Test-week rows with all 26 features, computed by training's own code."""
    if CACHE_FILE.exists():
        print(f"Using cached training answers: {CACHE_FILE.relative_to(PROJECT_ROOT)}")
        return pd.read_pickle(CACHE_FILE)

    print("Running training's feature code once (takes a few minutes; cached afterwards)...")
    _, _, test_week = lightgbm_tuned.load_data()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    test_week.to_pickle(CACHE_FILE)
    return test_week


def choose_transactions(answers: pd.DataFrame) -> pd.DataFrame:
    incident = answers[answers["route_flagged_down"] == 1]
    quiet = answers[answers["hour_of_day"].isin(QUIET_HOURS)]

    picked = pd.concat(
        [
            answers.sample(ORDINARY_ROWS, random_state=SEED),
            incident.sample(min(INCIDENT_ROWS, len(incident)), random_state=SEED),
            quiet.sample(min(QUIET_ROWS, len(quiet)), random_state=SEED),
        ]
    )
    return picked.drop_duplicates("txn_id").sort_values("txn_id")


def same_value(expected, actual) -> bool:
    expected_missing = pd.isna(expected)
    actual_missing = actual is None or pd.isna(actual)
    if expected_missing or actual_missing:
        return expected_missing and actual_missing
    return math.isclose(
        float(expected), float(actual), rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE
    )


def difference(expected, actual) -> tuple[float, float]:
    """(absolute, relative) difference, or (nan, nan) if one side is missing."""
    if pd.isna(expected) or actual is None or pd.isna(actual):
        return math.nan, math.nan
    absolute = abs(float(actual) - float(expected))
    relative = absolute / abs(float(expected)) if float(expected) != 0 else math.inf
    return absolute, relative


def main() -> int:
    os.chdir(PROJECT_ROOT)  # training code reads its CSV by relative path

    features = joblib.load(MODEL_PATH)["numeric_features"]
    answers = load_training_answers()
    picked = choose_transactions(answers)

    print("Loading history sources...")
    history = HistoryBuilder()

    mismatch_count = {f: 0 for f in features}
    worst = {f: (0.0, 0.0) for f in features}
    first_mismatch = {}
    gap_differences = {f: [] for f in KNOWN_GAPS}

    print(f"Checking {len(picked)} transactions...")
    for n, txn in enumerate(picked.itertuples(index=False), start=1):
        served = adapter.serving_features(history.history_before(txn), txn)

        for feature in features:
            expected = getattr(txn, feature)
            actual = served.get(feature)
            matched = same_value(expected, actual)
            absolute, relative = (0.0, 0.0) if matched else difference(expected, actual)

            if feature in KNOWN_GAPS:
                gap_differences[feature].append(relative if KNOWN_GAPS[feature].relative else absolute)

            if matched:
                continue
            mismatch_count[feature] += 1
            if not math.isnan(absolute) and absolute > worst[feature][0]:
                worst[feature] = (absolute, relative)
            first_mismatch.setdefault(feature, (txn, expected, actual))

        if n % 100 == 0:
            print(f"  {n} / {len(picked)}")

    status = {}
    gap_median = {}
    for feature in features:
        if feature in KNOWN_GAPS:
            gap_median[feature] = float(np.nanmedian(gap_differences[feature]))
            within = gap_median[feature] <= KNOWN_GAPS[feature].limit
            status[feature] = "KNOWN GAP" if within else "GAP TOO BIG"
        else:
            status[feature] = "PASS" if mismatch_count[feature] == 0 else "FAIL"

    print_report(features, picked, status, mismatch_count, worst, gap_median, first_mismatch)
    failed = any(s in ("FAIL", "GAP TOO BIG") for s in status.values())
    return 1 if failed else 0


def print_report(features, picked, status, mismatch_count, worst, gap_median, first_mismatch) -> None:
    total = len(picked)
    line = "-" * 96

    print()
    print("FEATURE PARITY REPORT")
    print(f"serving code under test : {adapter.NAME}")
    print(f"transactions checked    : {total}")
    print()
    print(f"{'feature':<36}{'mismatches':>14}{'worst abs diff':>16}{'worst rel diff':>16}{'result':>14}")
    print(line)

    for feature in features:
        misses = mismatch_count[feature]
        absolute, relative = worst[feature]
        abs_text = "-" if misses == 0 else ("NaN vs value" if absolute == 0 else f"{absolute:.6g}")
        rel_text = "-" if misses == 0 or absolute == 0 else f"{relative:.2%}"
        print(f"{feature:<36}{f'{misses} / {total}':>14}{abs_text:>16}{rel_text:>16}{status[feature]:>14}")

    print(line)
    counts = {s: sum(1 for f in features if status[f] == s) for s in ("PASS", "KNOWN GAP", "FAIL", "GAP TOO BIG")}
    print(
        f"{counts['PASS']} PASS, {counts['KNOWN GAP']} KNOWN GAP, "
        f"{counts['FAIL']} FAIL, {counts['GAP TOO BIG']} GAP TOO BIG  (of {len(features)} features)"
    )

    print()
    print("KNOWN GAPS")
    for feature, gap in KNOWN_GAPS.items():
        unit = "of training value" if gap.relative else "absolute"
        shown = f"{gap_median[feature]:.2%}" if gap.relative else f"{gap_median[feature]:.4f}"
        allowed = f"{gap.limit:.0%}" if gap.relative else f"{gap.limit}"
        print(f"  {feature}: median difference {shown} ({unit}), allowed {allowed} -> {status[feature]}")
        print(f"      {gap.reason}")

    real_failures = [f for f in features if status[f] in ("FAIL", "GAP TOO BIG")]
    if real_failures:
        print()
        print("FIRST MISMATCH FOR EACH FAILING FEATURE")
        for feature in real_failures:
            txn, expected, actual = first_mismatch[feature]
            print(f"  {feature}")
            print(
                f"      txn {txn.txn_id}  minute {txn.minute}  route {txn.route_id}  "
                f"bank {txn.bank}  network {txn.network}"
            )
            print(f"      training = {expected!r:<24}  serving = {actual!r}")


if __name__ == "__main__":
    sys.exit(main())
