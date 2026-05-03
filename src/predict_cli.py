import argparse
import json
import os
import joblib
import pandas as pd

MODEL_PATH = os.environ.get("MODEL_PATH", "models/delivery_time_model.pkl")

def load_model(path: str):
    return joblib.load(path)

def predict_one(model, distance_km, items_count, is_peak_hour, traffic_level):
    X = pd.DataFrame([{
        "distance_km": distance_km,
        "items_count": items_count,
        "is_peak_hour": is_peak_hour,
        "traffic_level": traffic_level
    }])
    return float(model.predict(X)[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_km", type=float, required=True)
    parser.add_argument("--items_count", type=int, required=True)
    parser.add_argument("--is_peak_hour", type=int, required=True)
    parser.add_argument("--traffic_level", type=int, required=True)

    args = parser.parse_args()

    model = load_model(MODEL_PATH)

    pred = predict_one(
        model,
        args.distance_km,
        args.items_count,
        args.is_peak_hour,
        args.traffic_level
    )

    print(json.dumps({"delivery_time_min": round(pred, 2)}, indent=2))

if __name__ == "__main__":
    main()