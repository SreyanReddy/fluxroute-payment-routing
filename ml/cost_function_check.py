import joblib
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "ml")
from lightgbm_tuned import load_data, prepare_xy

# ============================================================
# COST ASSUMPTIONS — placeholders, tune/justify these in your README
# ============================================================
FAILURE_COST_FIXED = 50.0          # flat friction/retry cost per missed failure
FAILURE_COST_PCT_OF_AMOUNT = 0.02  # + 2% of transaction amount at risk
MISROUTE_PENALTY = 5.0             # cost of a false alarm (routed away needlessly)
# route_cost_percent (already in the data) = fee charged on a successful txn

artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
calibrator = joblib.load("models/isotonic_calibrator_v2.joblib")

train_df, validation_df, test_df = load_data()


def compute_total_cost(df, threshold):
    X, y = prepare_xy(df)
    probs = calibrator.predict(model.predict_proba(preprocessor.transform(X))[:, 1])
    failure_probs = 1.0 - probs
    predicted_failure = failure_probs >= threshold

    y_np = y.to_numpy()
    amount = df["amount"].to_numpy()
    route_fee = amount * (df["route_cost_percent"].to_numpy() / 100.0)

    actual_failure = (y_np == 0)

    fn_mask = actual_failure & ~predicted_failure    # missed failure
    fp_mask = ~actual_failure & predicted_failure     # forgone a good route
    tn_mask = ~actual_failure & ~predicted_failure    # correctly processed -> revenue
    # tp_mask (correctly caught failure): cost 0, no fee either way

    cost = 0.0
    cost += fn_mask.sum() * FAILURE_COST_FIXED + (amount[fn_mask] * FAILURE_COST_PCT_OF_AMOUNT).sum()
    cost += route_fee[fp_mask].sum()      # opportunity cost = the fee we didn't get to earn
    cost -= route_fee[tn_mask].sum()      # revenue earned, subtract from cost

    return cost, cost / len(df)


thresholds = np.round(np.arange(0.02, 0.55, 0.02), 3)
results = []
for t in thresholds:
    total, per_txn = compute_total_cost(validation_df, t)
    results.append({"threshold": t, "validation_total_cost": round(total, 2), "cost_per_txn": round(per_txn, 4)})

results_df = pd.DataFrame(results).sort_values("cost_per_txn")
print(results_df.to_string(index=False))

best_threshold = results_df.iloc[0]["threshold"]
print(f"\nBest validation threshold by cost: {best_threshold}")

test_total, test_per_txn = compute_total_cost(test_df, best_threshold)
print(f"\nFINAL TEST COST at threshold {best_threshold}")
print(f"Total cost: {test_total:.2f}")
print(f"Cost per transaction: {test_per_txn:.4f}")