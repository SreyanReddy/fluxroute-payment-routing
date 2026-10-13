import sys
import pandas as pd
import joblib

sys.path.insert(0, "ml")
sys.path.insert(0, "dataset-generator-routing")

from lightgbm_tuned import (
    load_data,
    add_route_failure_rate_1m,
    add_time_since_last_bank_route_success,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from feature_extraction3 import (
    calculate_route_features,
    calculate_bank_route_features,
    calculate_training_baselines,
    add_time_features,
    ROUTE_CONFIG,
    TRAIN_FRACTION,
)
from routes import ROUTES as ROUTE_DEFS

ROUTE_ELIGIBILITY = {
    r.route_id: (set(r.supported_payment_methods), set(r.supported_networks))
    for r in ROUTE_DEFS
}
ALL_ROUTE_IDS = [r.route_id for r in ROUTE_DEFS]

RAW_PATH = "dataset-generator-routing/payment_dataset_actual_4week_v2.csv"
ROUTE_METRICS_PATH = "dataset-generator-routing/route_metrics_4week_v2.csv"

# ------------------------------------------------------------------
# Load model + calibrator (v2)
# ------------------------------------------------------------------
artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
calibrator = joblib.load("models/isotonic_calibrator_v2.joblib")

# ------------------------------------------------------------------
# Real historical data + aggregate route metrics
# ------------------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
raw = add_time_features(raw)
route_metrics = pd.read_csv(ROUTE_METRICS_PATH)

total_minutes = int(raw["minute"].max()) + 1
train_minutes = int(total_minutes * TRAIN_FRACTION)

base_stats = calculate_training_baselines(raw, train_minutes)
route_baselines = dict(zip(base_stats["route_id"], base_stats["route_base_success_rate"]))

route_features_full = calculate_route_features(route_metrics, total_minutes)
bank_route_features_full = calculate_bank_route_features(raw, route_baselines, total_minutes)

route_1m = add_route_failure_rate_1m(raw.copy())[
    ["route_id", "minute", "route_failure_rate_1m"]
].drop_duplicates(["route_id", "minute"])

bank_route_time = add_time_since_last_bank_route_success(raw.copy())[
    ["bank", "route_id", "minute", "time_since_last_bank_route_success"]
].drop_duplicates(["bank", "route_id", "minute"])

# ------------------------------------------------------------------
# Route-independent features: reuse what training already computed
# for the real chosen route (bank/network-level, not route-specific)
# ------------------------------------------------------------------
_, _, test_df = load_data()

txn_level = test_df[[
    "txn_id", "minute", "day_in_week", "hour_of_day",
    "amount", "bank", "network", "payment_method", "merchant", "device", "risk",
    "bank_failure_rate_5m", "network_failure_rate_5m", "amount_to_bank_avg_ratio",
    "route_id", "success",
]].rename(columns={"route_id": "actual_route_id", "success": "actual_success"})

# ------------------------------------------------------------------
# Expand: one row per (test transaction x candidate route)
# ------------------------------------------------------------------
candidates = pd.concat(
    [txn_level.assign(route_id=r) for r in ALL_ROUTE_IDS],
    ignore_index=True,
)

eligible_mask = pd.Series(False, index=candidates.index)
for route_id, (pm_set, net_set) in ROUTE_ELIGIBILITY.items():
    on_route = candidates["route_id"] == route_id
    eligible_mask |= (
        on_route
        & candidates["payment_method"].isin(pm_set)
        & candidates["network"].isin(net_set)
    )
candidates = candidates[eligible_mask].copy()

# Static per-route features
base_stats_map = base_stats.set_index("route_id")
candidates["route_base_success_rate"] = candidates["route_id"].map(base_stats_map["route_base_success_rate"])
candidates["route_base_latency_ms"] = candidates["route_id"].map(base_stats_map["route_base_latency_ms"])
candidates["route_cost_percent"] = candidates["route_id"].map(lambda r: ROUTE_CONFIG[r]["route_cost_percent"])

# Real-time per-route features, looked up per candidate route
candidates = candidates.merge(route_features_full, on=["route_id", "minute"], how="left")
candidates = candidates.merge(bank_route_features_full, on=["bank", "route_id", "minute"], how="left")
candidates = candidates.merge(route_1m, on=["route_id", "minute"], how="left")
candidates = candidates.merge(bank_route_time, on=["bank", "route_id", "minute"], how="left")

# Derived engineered features (pure arithmetic, safe to recompute per row)
candidates["route_failure_rate_5m"] = 1.0 - candidates["route_success_rate_5m"]
candidates["route_failure_rate_15m"] = 1.0 - candidates["route_success_rate_15m"]
candidates["route_success_drop_5m"] = candidates["route_base_success_rate"] - candidates["route_success_rate_5m"]
candidates["route_success_drop_15m"] = candidates["route_base_success_rate"] - candidates["route_success_rate_15m"]
candidates["route_latency_ratio"] = candidates["route_avg_latency_5m"] / (candidates["route_base_latency_ms"] + 1e-6)
candidates["bank_route_gap"] = candidates["bank_route_success_rate_15m"] - candidates["route_success_rate_15m"]
candidates["route_stress"] = candidates["route_utilization"] * candidates["route_failure_rate_15m"]

# ------------------------------------------------------------------
# Score every candidate route
# ------------------------------------------------------------------
X_candidates = candidates[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
raw_probs = model.predict_proba(preprocessor.transform(X_candidates))[:, 1]
candidates["predicted_success_prob"] = calibrator.predict(raw_probs)

# Outcome proxy for whichever route a policy picks -- now backed by
# aggregate binomial draws (route_metrics), much lower noise than the
# v1 sampled-row-derived rolling rate.
candidates["outcome_proxy"] = candidates["route_success_rate_5m"]

# ------------------------------------------------------------------
# Policy: Model argmax among eligible routes
# ------------------------------------------------------------------
model_choice = (
    candidates.sort_values("predicted_success_prob", ascending=False)
    .groupby("txn_id", as_index=False)
    .first()
)

# ------------------------------------------------------------------
# Policy: Round robin, cycling R1..R4 skipping ineligible routes
# ------------------------------------------------------------------
eligible_by_txn = candidates.groupby("txn_id")["route_id"].apply(list).to_dict()
rr_choice_map = {}
rotation = 0
for txn_id in sorted(eligible_by_txn.keys()):
    eligible = eligible_by_txn[txn_id]
    for _ in range(len(ALL_ROUTE_IDS)):
        candidate_route = ALL_ROUTE_IDS[rotation % len(ALL_ROUTE_IDS)]
        rotation += 1
        if candidate_route in eligible:
            rr_choice_map[txn_id] = candidate_route
            break

candidates["rr_choice"] = candidates["txn_id"].map(rr_choice_map)
rr_rows = candidates[candidates["route_id"] == candidates["rr_choice"]]

# ------------------------------------------------------------------
# Anchor: what actually happened under the real weighted-random router
# ------------------------------------------------------------------
actual_success_rate = txn_level["actual_success"].mean()

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
print(f"Test transactions evaluated: {txn_level['txn_id'].nunique():,}")
print(f"Candidate (txn, route) rows after eligibility filter: {len(candidates):,}\n")

print(f"ACTUAL (real weighted-random router)  success rate: {actual_success_rate:.4%}")
print(f"ROUND-ROBIN (proxy outcome)           success rate: {rr_rows['outcome_proxy'].mean():.4%}")
print(f"MODEL ARGMAX (proxy outcome)          success rate: {model_choice['outcome_proxy'].mean():.4%}\n")

delta = model_choice["outcome_proxy"].mean() - rr_rows["outcome_proxy"].mean()
print(f"Model vs Round-Robin delta: {delta:+.4%}")

print("\nEligible routes per network (sanity check -- watch for RuPay):")
print(candidates.groupby("network")["route_id"].value_counts())