import joblib
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "ml")
from lightgbm_tuned import load_data, prepare_xy

artifact = joblib.load("models/lightgbm_tuned.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
saved_threshold = artifact.get("threshold", 0.5)

_, _, test_df = load_data()
X_test, y_test = prepare_xy(test_df)

X_test_transformed = preprocessor.transform(X_test)
success_probabilities = model.predict_proba(X_test_transformed)[:, 1]

bins = np.arange(0, 1.05, 0.1)
bin_idx = np.digitize(success_probabilities, bins) - 1

rows = []
for b in range(len(bins) - 1):
    mask = bin_idx == b
    n = mask.sum()
    if n == 0:
        continue
    actual_success_rate = y_test.to_numpy()[mask].mean()
    mean_predicted_prob = success_probabilities[mask].mean()
    rows.append({
        "bin": f"[{bins[b]:.1f}-{bins[b+1]:.1f})",
        "n": n,
        "mean_predicted_prob": round(mean_predicted_prob, 4),
        "actual_success_rate": round(actual_success_rate, 4),
        "gap": round(mean_predicted_prob - actual_success_rate, 4),
    })

result = pd.DataFrame(rows)
print(f"Saved validation threshold: {saved_threshold:.6f}\n")
print(result.to_string(index=False))