import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "ml")
from lightgbm_tuned import load_data, prepare_xy

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

FAILURE_COST_FIXED = 50.0
FAILURE_COST_PCT_OF_AMOUNT = 0.02
TARGET_FAILURE_RECALL = 0.50

artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
calibrator = joblib.load("models/isotonic_calibrator_v2.joblib")

train_df, validation_df, test_df = load_data()


def score_dataset(df):
    X, y = prepare_xy(df)
    probs = calibrator.predict(model.predict_proba(preprocessor.transform(X))[:, 1])
    return probs, y.to_numpy(), df["amount"].to_numpy(), df["route_cost_percent"].to_numpy() / 100.0


def evaluate_threshold(probs, y, amount, route_fee_pct, threshold):
    failure_probs = 1.0 - probs
    predicted_failure = failure_probs >= threshold
    predictions = (~predicted_failure).astype(int)

    actual_failure = (y == 0)
    fn_mask = actual_failure & ~predicted_failure
    fp_mask = ~actual_failure & predicted_failure
    tn_mask = ~actual_failure & ~predicted_failure

    route_fee = amount * route_fee_pct
    cost = (
        fn_mask.sum() * FAILURE_COST_FIXED
        + (amount[fn_mask] * FAILURE_COST_PCT_OF_AMOUNT).sum()
        + route_fee[fp_mask].sum()
        - route_fee[tn_mask].sum()
    )

    return {
        "threshold": threshold,
        "cost_per_txn": cost / len(y),
        "failure_precision": precision_score(y, predictions, pos_label=0, zero_division=0),
        "failure_recall": recall_score(y, predictions, pos_label=0, zero_division=0),
        "failure_f1": f1_score(y, predictions, pos_label=0, zero_division=0),
    }


# ------------------------------------------------------------------
# Sweep on VALIDATION -- cost and classification metrics together
# ------------------------------------------------------------------
val_probs, val_y, val_amount, val_fee_pct = score_dataset(validation_df)

thresholds = np.round(np.arange(0.02, 0.55, 0.02), 3)
sweep = pd.DataFrame([
    evaluate_threshold(val_probs, val_y, val_amount, val_fee_pct, t)
    for t in thresholds
])

print("VALIDATION SWEEP")
print(sweep.to_string(index=False))

# Selection A: pure cost-optimal
cost_optimal = sweep.loc[sweep["cost_per_txn"].idxmin()]

# Selection B: cheapest threshold that still hits the recall floor
eligible = sweep[sweep["failure_recall"] >= TARGET_FAILURE_RECALL]
constrained_optimal = (
    eligible.loc[eligible["cost_per_txn"].idxmin()] if not eligible.empty else None
)

print(f"\nSelection A -- pure cost-optimal:\n{cost_optimal}")
if constrained_optimal is not None:
    print(f"\nSelection B -- cheapest threshold with recall >= {TARGET_FAILURE_RECALL:.0%}:\n{constrained_optimal}")
else:
    print(f"\nNo threshold in range achieves recall >= {TARGET_FAILURE_RECALL:.0%}")

# ------------------------------------------------------------------
# Final test-set evaluation -- both candidates, single touch
# ------------------------------------------------------------------
test_probs, test_y, test_amount, test_fee_pct = score_dataset(test_df)

for label, row in [("A: pure cost-optimal", cost_optimal), ("B: constrained cost-optimal", constrained_optimal)]:
    if row is None:
        continue
    t = row["threshold"]
    result = evaluate_threshold(test_probs, test_y, test_amount, test_fee_pct, t)
    predictions = (~((1.0 - test_probs) >= t)).astype(int)

    print(f"\n{'='*60}\nTEST RESULTS -- {label} (threshold={t})\n{'='*60}")
    print(f"Cost per txn: {result['cost_per_txn']:.4f}")
    print(f"Failure Precision: {result['failure_precision']:.4f}")
    print(f"Failure Recall:    {result['failure_recall']:.4f}")
    print(f"Failure F1:        {result['failure_f1']:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_y, predictions))
    print("\nClassification Report:")
    print(classification_report(test_y, predictions))