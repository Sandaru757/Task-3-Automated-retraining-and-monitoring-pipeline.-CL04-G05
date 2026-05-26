import os
import json
import joblib
import numpy as np

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Paths
# -----------------------------
X_TRAIN_PATH = "artifacts/data/X_train.npy"
Y_TRAIN_PATH = "artifacts/data/y_train.npy"

MODEL_DIR = "artifacts/models"
METRICS_DIR = "artifacts/metrics"

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
TRAINING_HISTORY_PATH = os.path.join(METRICS_DIR, "training_history.json")


def create_folders():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def main():
    print("Starting model training...")

    create_folders()

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(f"Missing file: {X_TRAIN_PATH}")

    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(f"Missing file: {Y_TRAIN_PATH}")

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)

    mae = mean_absolute_error(y_train, train_predictions)
    mse = mean_squared_error(y_train, train_predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_train, train_predictions)

    joblib.dump(model, MODEL_PATH)

    training_history = {
        "model_type": "RandomForestRegressor",
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": int(X_train.shape[0]),
        "number_of_features": int(X_train.shape[1]),
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "random_state": 42
        },
        "training_metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2_score": float(r2)
        }
    }

    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(training_history, f, indent=4)

    print("Model training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training MAE: {mae:.4f}")
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Training R2 Score: {r2:.4f}")


if __name__ == "__main__":
    main()