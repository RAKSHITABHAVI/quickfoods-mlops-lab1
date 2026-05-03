import os
import json
import mlflow
import pandas as pd
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 🔥 Registry config
MODEL_NAME = "quickfoods-delivery-predictor"
MODEL_ALIAS = "champion"

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")

app = FastAPI(
    title="QuickFoods Delivery API (Registry)",
    version="3.0"
)

os.makedirs(LOG_DIR, exist_ok=True)

# 🔥 Load model from MLflow Registry
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
print(f"Loading model from: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully")

# Input schema
class DeliveryRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    items_count: int = Field(..., gt=0)
    is_peak_hour: int = Field(..., ge=0, le=1)
    traffic_level: int = Field(..., ge=1, le=3)

# Output schema
class PredictionResponse(BaseModel):
    delivery_time_min: float

# Logging
def log_prediction(input_data, prediction):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": prediction,
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "alias": MODEL_ALIAS
    }

# Prediction API
@app.post("/predict", response_model=PredictionResponse)
def predict(request: DeliveryRequest):
    try:
        input_dict = request.dict()
        df = pd.DataFrame([input_dict])

        pred = round(float(model.predict(df)[0]), 2)

        log_prediction(input_dict, pred)

        return {"delivery_time_min": pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))