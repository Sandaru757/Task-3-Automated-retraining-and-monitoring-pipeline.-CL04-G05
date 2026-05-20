"""
Monitoring script - placeholder for Member 3.
Drift detection using Kolmogorov-Smirnov test.
"""
import json
import os
from datetime import datetime
import numpy as np
from scipy import stats
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

ARTIFACTS_DATA = "artifacts/data"
ARTIFACTS_MODELS = "artifacts/models"
ARTIFACTS_METRICS = "artifacts/metrics"
REPORTS = "reports"

os.makedirs(REPORTS, exist_ok=True)
os.makedirs(ARTIFACTS_METRICS, exist_ok=True)

X_train = np.load(f"{ARTIFACTS_DATA}/X_train.npy")
X_test = np.load(f"{ARTIFACTS_DATA}/X_test.npy")
y_test = np.load(f"{ARTIFACTS_DATA}/y_test.npy")
model = joblib.load(f"{ARTIFACTS_MODELS}/model.pkl")

drift_results = []
DRIFT_THRESHOLD = 0.05

for i in range(X_train.shape[1]):
    statistic, p_value = stats.ks_2samp(X_train[:, i], X_test[:, i])
    drift_results.append({
        "feature_index": i,
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < DRIFT_THRESHOLD)
    })

n_drifted = sum(1 for r in drift_results if r["drift_detected"])

drift_report = {
    "timestamp": datetime.now().isoformat(),
    "method": "Kolmogorov-Smirnov test",
    "threshold_p_value": DRIFT_THRESHOLD,
    "n_features": len(drift_results),
    "n_drifted_features": n_drifted,
    "drift_share": n_drifted / len(drift_results),
    "feature_results": drift_results
}

with open(f"{REPORTS}/drift_report.json", "w") as f:
    json.dump(drift_report, f, indent=2)

y_pred = model.predict(X_test)
monitoring_metrics = {
    "timestamp": datetime.now().isoformat(),
    "MAE": float(mean_absolute_error(y_test, y_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "drift_share": n_drifted / len(drift_results)
}

with open(f"{ARTIFACTS_METRICS}/monitoring_metrics.json", "w") as f:
    json.dump(monitoring_metrics, f, indent=2)

print(f"Monitoring - {n_drifted}/{len(drift_results)} features drifted")
