"""
Training script - placeholder for Member 2.
Trains a simple regression model. Uses sklearn instead of TensorFlow for speed.
"""
import json
import os
from datetime import datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ARTIFACTS_DATA = "artifacts/data"
ARTIFACTS_MODELS = "artifacts/models"
ARTIFACTS_METRICS = "artifacts/metrics"
ARTIFACTS_METADATA = "artifacts/metadata"

os.makedirs(ARTIFACTS_MODELS, exist_ok=True)
os.makedirs(ARTIFACTS_METRICS, exist_ok=True)
os.makedirs(ARTIFACTS_METADATA, exist_ok=True)

X_train = np.load(f"{ARTIFACTS_DATA}/X_train.npy")
y_train = np.load(f"{ARTIFACTS_DATA}/y_train.npy")

print(f"Training on {X_train.shape[0]} samples, {X_train.shape[1]} features")

model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
train_mae = float(mean_absolute_error(y_train, train_pred))

joblib.dump(model, f"{ARTIFACTS_MODELS}/model.pkl")

history = {
    "train_mae": train_mae,
    "n_estimators": 50,
    "max_depth": 10,
    "n_train_samples": int(X_train.shape[0])
}
with open(f"{ARTIFACTS_METRICS}/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

with open(f"{ARTIFACTS_METADATA}/model_version.txt", "w") as f:
    f.write(f"v1.0 - {datetime.now().isoformat()}\n")
with open(f"{ARTIFACTS_METADATA}/last_retrain.txt", "w") as f:
    f.write(datetime.now().isoformat() + "\n")

print(f"Training complete. Train MAE: {train_mae:.2f}")
