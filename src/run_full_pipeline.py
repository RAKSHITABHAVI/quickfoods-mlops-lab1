import os
import sys
import subprocess
import time
import requests

API_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_URL}/predict"
HEALTH_URL = f"{API_URL}/health"
LOG_PATH = "logs/predictions.jsonl"


def run_script(name, script):
    print(f"\n{'='*60}")
    print(f"STAGE: {name}")
    print(f"{'='*60}")

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"\n❌ FAILED: {script}")
        sys.exit(1)

    print(f"✅ {name} DONE\n")


def start_api():
    print("\n🚀 Starting API...\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--port", "8000"]
    )

    for _ in range(15):
        time.sleep(1)
        try:
            if requests.get(HEALTH_URL).status_code == 200:
                print("✅ API is running\n")
                return proc
        except:
            pass

    print("❌ API failed")
    sys.exit(1)


def send_requests():
    print("\n📡 Sending sample requests...\n")

    import random

    for i in range(10):
        payload = {
            "distance_km": round(random.uniform(1, 10), 1),
            "items_count": random.randint(1, 5),
            "is_peak_hour": random.choice([0, 1]),
            "traffic_level": random.choice([1, 2, 3]),
        }

        r = requests.post(PREDICT_URL, json=payload)
        print(f"{i+1}: {r.json()}")

    print("\n✅ Requests done\n")


def main():

    print("🔥 FULL MLOPS PIPELINE STARTED\n")

    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    # 1. Train
    run_script("Train Model", "src/train.py")

    # 2. MLflow tracking
    run_script("MLflow Training", "src/train_multi_metrics_with_mlflow.py")

    # 3. Hyperparameter tuning
    run_script("Hyperparameter Tuning", "src/train_hyperparameter_tuning.py")

    # 4. Promote model
    run_script("Promote Model", "src/promote_model.py")

    # 5. Start API
    api = start_api()

    try:
        # 6. Send traffic
        send_requests()
    finally:
        print("\n🛑 Stopping API...\n")
        api.terminate()

    # 7. Retrain
    run_script("Retraining", "src/retrain.py")

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()