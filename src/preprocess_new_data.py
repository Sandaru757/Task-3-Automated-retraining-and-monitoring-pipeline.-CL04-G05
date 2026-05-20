"""
Preprocessing script - placeholder for Member 2.
Reads raw power consumption data, splits into train/test, saves artifacts.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/power_consumption.csv"
ARTIFACTS_DATA = "artifacts/data"
ARTIFACTS_PREPROCESSING = "artifacts/preprocessing"

os.makedirs(ARTIFACTS_DATA, exist_ok=True)
os.makedirs(ARTIFACTS_PREPROCESSING, exist_ok=True)

print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

df = df.dropna()

target_col = "Zone 1 Power Consumption"
if target_col not in df.columns:
    target_col = df.select_dtypes(include=[np.number]).columns[-1]
    print(f"Using fallback target column: {target_col}")

features_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
target = df[target_col].values

feature_columns = features_df.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    features_df.values, target, test_size=0.2, shuffle=False
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.save(f"{ARTIFACTS_DATA}/X_train.npy", X_train_scaled)
np.save(f"{ARTIFACTS_DATA}/y_train.npy", y_train)
np.save(f"{ARTIFACTS_DATA}/X_test.npy", X_test_scaled)
np.save(f"{ARTIFACTS_DATA}/y_test.npy", y_test)
joblib.dump(scaler, f"{ARTIFACTS_PREPROCESSING}/scaler.pkl")

with open(f"{ARTIFACTS_PREPROCESSING}/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

print(f"Preprocessing complete. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
