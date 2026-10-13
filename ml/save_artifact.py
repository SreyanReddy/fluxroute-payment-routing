import joblib

artifact = joblib.load("models/lightgbm_tuned_v2.joblib")
artifact["threshold"] = 0.18
joblib.dump(artifact, "models/lightgbm_tuned_v2.joblib")
print("Threshold corrected to 0.18")