import os
import json
import time
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split(df: pd.DataFrame):
    X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
    y = df["delivery_time_min"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

def train_and_log(model_name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)

        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate(y_test, preds)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        ensure_dir(MODEL_DIR)
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "model")

        print(f"{model_name} → MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.2f}")

        return {"model_name": model_name, **metrics}

def main():
    print("=== Lab 3: Multi-model tracking ===")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split(df)

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    results.append(train_and_log("LinearRegression", LinearRegression(), {}, X_train, X_test, y_train, y_test))

    results.append(train_and_log(
        "RandomForest",
        RandomForestRegressor(n_estimators=100, random_state=42),
        {"n_estimators": 100},
        X_train, X_test, y_train, y_test
    ))

    results.append(train_and_log(
        "GradientBoosting",
        GradientBoostingRegressor(random_state=42),
        {},
        X_train, X_test, y_train, y_test
    ))

    best = sorted(results, key=lambda x: x["mae"])[0]

    print("\nBest model (lowest MAE):")
    print(best)

if __name__ == "__main__":
    main()