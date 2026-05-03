import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/delivery_times.csv"
REGISTERED_NAME = "quickfoods-delivery-predictor"

df = pd.read_csv(DATA_PATH)

X = df[["distance_km", "items_count", "is_peak_hour", "traffic_level"]]
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

mlflow.set_experiment("quickfoods-delivery-time")

with mlflow.start_run(run_name="RandomForest-V2"):

    model = RandomForestRegressor(n_estimators=200, max_depth=8)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"

    mlflow.register_model(model_uri, REGISTERED_NAME)

    print("✅ Version 2 registered")