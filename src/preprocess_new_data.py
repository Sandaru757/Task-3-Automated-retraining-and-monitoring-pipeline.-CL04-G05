import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/power_consumption.csv"

ARTIFACT_DATA_DIR = "artifacts/data"
ARTIFACT_PREPROCESSING_DIR = "artifacts/preprocessing"

X_TRAIN_PATH = os.path.join(ARTIFACT_DATA_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(ARTIFACT_DATA_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(ARTIFACT_DATA_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(ARTIFACT_DATA_DIR, "y_test.npy")

SCALER_PATH = os.path.join(ARTIFACT_PREPROCESSING_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACT_PREPROCESSING_DIR, "feature_columns.json")


def create_folders():
    os.makedirs(ARTIFACT_DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_PREPROCESSING_DIR, exist_ok=True)


def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def add_datetime_features(df):
    """
    Adds useful time-based features if a datetime column exists.
    Works with Tetouan dataset style DateTime column.
    """
    possible_datetime_cols = ["DateTime", "Datetime", "date_time", "datetime", "Date_Time"]

    datetime_col = None
    for col in possible_datetime_cols:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is not None:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

        df["hour"] = df[datetime_col].dt.hour
        df["day"] = df[datetime_col].dt.day
        df["month"] = df[datetime_col].dt.month
        df["weekday"] = df[datetime_col].dt.weekday

        df = df.drop(columns=[datetime_col])

    return df


def main():
    print("Starting preprocessing...")

    create_folders()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)
    df = add_datetime_features(df)

    print(f"Dataset shape before cleaning: {df.shape}")

    # Remove rows with missing values
    df = df.dropna()

    # Target column for Tetouan power consumption dataset
    possible_targets = [
        "Zone_1_Power_Consumption",
        "Zone_1_Power_Consumption_",
        "Zone_1",
        "Power_Consumption"
    ]

    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            "Target column not found. Expected something like "
            "'Zone_1_Power_Consumption'. Available columns are: "
            f"{list(df.columns)}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Keep only numeric columns for model training
    X = X.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("No numeric feature columns found after preprocessing.")

    feature_columns = list(X.columns)

    # Time-series style split: no shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    np.save(X_TRAIN_PATH, X_train_scaled)
    np.save(X_TEST_PATH, X_test_scaled)
    np.save(Y_TRAIN_PATH, y_train.to_numpy())
    np.save(Y_TEST_PATH, y_test.to_numpy())

    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=4)

    print("Preprocessing completed successfully.")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Target column: {target_col}")


if __name__ == "__main__":
    main()
