import joblib
import pandas as pd
import numpy as np
import sys
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "ml")
from lightgbm_tuned import load_data, prepare_xy

artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]

train_df, validation_df, test_df = load_data()

X_val, y_val = prepare_xy(validation_df)
X_test, y_test = prepare_xy(test_df)

val_probs = model.predict_proba(preprocessor.transform(X_val))[:, 1]
test_probs = model.predict_proba(preprocessor.transform(X_test))[:, 1]

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(val_probs, y_val)

test_probs_calibrated = calibrator.predict(test_probs)

joblib.dump(calibrator, "models/isotonic_calibrator_v2.joblib")

bins = np.arange(0, 1.05, 0.1)
bin_idx = np.digitize(test_probs_calibrated, bins) - 1
y_test_np = y_test.to_numpy()

rows = []
for b in range(len(bins) - 1):
    mask = bin_idx == b
    n = mask.sum()
    if n == 0:
        continue
    rows.append({
        "bin": f"[{bins[b]:.1f}-{bins[b+1]:.1f})",
        "n": n,
        "mean_predicted_prob": round(test_probs_calibrated[mask].mean(), 4),
        "actual_success_rate": round(y_test_np[mask].mean(), 4),
        "gap": round(test_probs_calibrated[mask].mean() - y_test_np[mask].mean(), 4),
    })

print(pd.DataFrame(rows).to_string(index=False))
print(f"\nROC-AUC unaffected by calibration (rank-preserving); "
      f"Brier before vs after:")
from sklearn.metrics import brier_score_loss
print(f"  Before: {brier_score_loss(y_test_np, test_probs):.6f}")
print(f"  After:  {brier_score_loss(y_test_np, test_probs_calibrated):.6f}")