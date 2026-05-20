"""
Evaluation script - placeholder for Member 2.
Computes MAE, RMSE, R-squared on the test set.
"""
import json
import os
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARTIFACTS_DATA = "artifacts/data"
ARTIFACTS_MODELS = "artifacts/models"
ARTIFACTS_METRICS = "artifacts/metrics"

os.makedirs(ARTIFACTS_METRICS, exist_ok=True)

model = joblib.load(f"{ARTIFACTS_MODELS}/model.pkl")
X_test = np.load(f"{ARTIFACTS_DATA}/X_test.npy")
y_test = np.load(f"{ARTIFACTS_DATA}/y_test.npy")

y_pred = model.predict(X_test)

mae = float(mean_absolute_error(y_test, y_pred))
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = float(r2_score(y_test, y_pred))

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "n_test_samples": int(len(y_test))
}

with open(f"{ARTIFACTS_METRICS}/evaluation_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
