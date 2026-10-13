import sys
sys.path.insert(0, "serving")
sys.path.insert(0, "ml")

import joblib
import pandas as pd
from lightgbm_tuned import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from feature_storing import build_route_features, ROUTE_STATIC_CONFIG

artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
model = artifact["model"]
preprocessor = artifact["preprocessor"]
calibrator = joblib.load("models/isotonic_calibrator_v2.joblib")

rows = [
    build_route_features(
        route_id=route_id, bank="HDFC", network="UPI", amount=2500.0,
        payment_method="UPI", merchant="Shopping", device="Android", risk="LOW",
    )
    for route_id in ROUTE_STATIC_CONFIG.keys()
]

df = pd.DataFrame(rows)
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

raw_probs = model.predict_proba(preprocessor.transform(X))[:, 1]
calibrated_probs = calibrator.predict(raw_probs)

for route_id, raw, calibrated in zip(df["route_id"], raw_probs, calibrated_probs):
    print(f"{route_id}: raw={raw:.10f}  calibrated={calibrated:.10f}")