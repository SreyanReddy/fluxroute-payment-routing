import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.insert(0, "ml")
from lightgbm_tuned import load_data, prepare_xy

BASE_FAILURE_COST_FIXED = 50.0
BASE_FAILURE_COST_PCT_OF_AMOUNT = 0.02

artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
calibrator = joblib.load("models/isotonic_calibrator_v2.joblib")

_, validation_df, _ = load_data()


def score_dataset(df):
    X, y = prepare_xy(df)
    probs = calibrator.predict(model.predict_proba(preprocessor.transform(X))[:, 1])
    return probs, y.to_numpy(), df["amount"].to_numpy(), df["route_cost_percent"].to_numpy() / 100.0


def evaluate_threshold(probs, y, amount, route_fee_pct, threshold, fc_fixed, fc_pct):
    failure_probs = 1.0 - probs
    predicted_failure = failure_probs >= threshold
    predictions = (~predicted_failure).astype(int)

    actual_failure = (y == 0)
    fn_mask = actual_failure & ~predicted_failure
    fp_mask = ~actual_failure & predicted_failure
    tn_mask = ~actual_failure & ~predicted_failure

    route_fee = amount * route_fee_pct
    cost = (
        fn_mask.sum() * fc_fixed
        + (amount[fn_mask] * fc_pct).sum()
        + route_fee[fp_mask].sum()
        - route_fee[tn_mask].sum()
    )

    return {
        "threshold": threshold,
        "cost_per_txn": cost / len(y),
        "failure_precision": precision_score(y, predictions, pos_label=0, zero_division=0),
        "failure_recall": recall_score(y, predictions, pos_label=0, zero_division=0),
    }


val_probs, val_y, val_amount, val_fee_pct = score_dataset(validation_df)
thresholds = np.round(np.arange(0.02, 0.55, 0.02), 3)

print(f"{'Cost mult.':>10} {'Best thresh':>12} {'Cost/txn':>10} {'Precision':>10} {'Recall':>9}")
for mult in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    fc_fixed = BASE_FAILURE_COST_FIXED * mult
    fc_pct = BASE_FAILURE_COST_PCT_OF_AMOUNT * mult
    sweep = pd.DataFrame([
        evaluate_threshold(val_probs, val_y, val_amount, val_fee_pct, t, fc_fixed, fc_pct)
        for t in thresholds
    ])
    best = sweep.loc[sweep["cost_per_txn"].idxmin()]
    print(f"{mult:>10.1f} {best['threshold']:>12.3f} {best['cost_per_txn']:>10.4f} {best['failure_precision']:>10.4f} {best['failure_recall']:>9.4f}")