import os
import json
import joblib
import numpy as np

from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Paths
# -----------------------------
X_TEST_PATH = "artifacts/data/X_test.npy"
Y_TEST_PATH = "artifacts/data/y_test.npy"

MODEL_PATH = "artifacts/models/model.pkl"

METRICS_DIR = "artifacts/metrics"
EVALUATION_METRICS_PATH = os.path.join(METRICS_DIR, "evaluation_metrics.json")


def create_folders():
    os.makedirs(METRICS_DIR, exist_ok=True)


def main():
    print("Starting model evaluation...")

    create_folders()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(f"Missing file: {X_TEST_PATH}")

    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError(f"Missing file: {Y_TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    evaluation_metrics = {
        "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_samples": int(X_test.shape[0]),
        "number_of_features": int(X_test.shape[1]),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2)
        }
    }

    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(evaluation_metrics, f, indent=4)

    print("Model evaluation completed successfully.")
    print(f"Evaluation metrics saved to: {EVALUATION_METRICS_PATH}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()