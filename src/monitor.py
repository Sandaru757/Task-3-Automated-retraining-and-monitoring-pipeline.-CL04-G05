import json
import os
import joblib
import numpy as np

from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ======================================
# PATHS
# ======================================
MODEL_PATH = "artifacts/models/model.pkl"

X_TEST_PATH = "artifacts/data/X_test.npy"
Y_TEST_PATH = "artifacts/data/y_test.npy"

DRIFT_REPORT_PATH = "reports/drift_report.json"

MONITORING_METRICS_PATH = (
    "artifacts/metrics/monitoring_metrics.json"
)


# ======================================
# LOAD MODEL
# ======================================
print("[INFO] Loading model...")

model = joblib.load(MODEL_PATH)


# ======================================
# LOAD DATA
# ======================================
print("[INFO] Loading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)


# ======================================
# MAKE PREDICTIONS
# ======================================
print("[INFO] Running predictions...")

y_pred = model.predict(X_test)


# ======================================
# PERFORMANCE METRICS
# ======================================
print("[INFO] Calculating metrics...")

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


# ======================================
# DRIFT DETECTION
# ======================================
print("[INFO] Detecting feature drift...")

# Split test data into:
# reference distribution + current distribution

midpoint = len(X_test) // 2

reference_data = X_test[:midpoint]
current_data = X_test[midpoint:]


feature_drift = {}

for i in range(X_test.shape[1]):

    ks_statistic, p_value = ks_2samp(
        reference_data[:, i],
        current_data[:, i]
    )

    drift_detected = p_value < 0.05

    feature_drift[f"feature_{i}"] = {
        "ks_statistic": float(ks_statistic),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected)
    }


# ======================================
# OVERALL DRIFT SUMMARY
# ======================================
drifted_features = sum(
    feature["drift_detected"]
    for feature in feature_drift.values()
)

total_features = X_test.shape[1]

drift_percentage = (
    drifted_features / total_features
)

overall_drift = {
    "total_features": int(total_features),
    "drifted_features": int(drifted_features),
    "drift_percentage": float(drift_percentage)
}


# ======================================
# CREATE OUTPUT DIRECTORIES
# ======================================
os.makedirs("reports", exist_ok=True)

os.makedirs(
    "artifacts/metrics",
    exist_ok=True
)


# ======================================
# SAVE DRIFT REPORT
# ======================================
print("[INFO] Saving drift report...")

drift_report = {
    "overall_drift": overall_drift,
    "feature_drift": feature_drift
}

with open(DRIFT_REPORT_PATH, "w") as f:
    json.dump(
        drift_report,
        f,
        indent=4
    )


# ======================================
# SAVE MONITORING METRICS
# ======================================
print("[INFO] Saving monitoring metrics...")

monitoring_metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
    "drift_summary": overall_drift
}

with open(MONITORING_METRICS_PATH, "w") as f:
    json.dump(
        monitoring_metrics,
        f,
        indent=4
    )


# ======================================
# FINISHED
# ======================================
print("[INFO] Monitoring complete.")

print(json.dumps(
    monitoring_metrics,
    indent=4
))