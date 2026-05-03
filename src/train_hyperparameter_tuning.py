import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from itertools import product as cartesian_product
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

DATA_PATH = "data/delivery_times.csv"
MODEL_DIR = "models"
EXPERIMENT_NAME = "quickfoods-delivery-time"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = ["distance_km", "items_count", "is_peak_hour", "traffic_level"]
TARGET = "delivery_time_min"

RF_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10],
}

GB_PARAM_GRID = {
    "n_estimators": [50, 100],
    "learning_rate": [0.1, 0.2],
    "max_depth": [3, 5],
}

def load_data(path):
    return pd.read_csv(path)

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

def run_trial(model_name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=str(params), nested=True):
        mlflow.log_param("model_name", model_name)

        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate(y_test, preds)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        return metrics["mae"]

def main():
    print("=== Lab 6: Hyperparameter Tuning ===")

    df = load_data(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="HyperparamSweep"):
        best_mae = float("inf")
        best_model = None

        # Grid Search RF
        for params in cartesian_product(*RF_PARAM_GRID.values()):
            param_dict = dict(zip(RF_PARAM_GRID.keys(), params))
            model = RandomForestRegressor(**param_dict, random_state=42)

            mae = run_trial("RandomForest", model, param_dict, X_train, X_test, y_train, y_test)

            if mae < best_mae:
                best_mae = mae
                best_model = ("RandomForest", param_dict)

        # Random Search GB
        for _ in range(4):
            param_dict = {k: np.random.choice(v) for k, v in GB_PARAM_GRID.items()}
            model = GradientBoostingRegressor(**param_dict, random_state=42)

            mae = run_trial("GradientBoosting", model, param_dict, X_train, X_test, y_train, y_test)

            if mae < best_mae:
                best_mae = mae
                best_model = ("GradientBoosting", param_dict)

        print("\nBest model:")
        print(best_model)
        print("Best MAE:", best_mae)

if __name__ == "__main__":
    main()