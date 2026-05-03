from mlflow.tracking import MlflowClient

MODEL_NAME = "quickfoods-delivery-predictor"

client = MlflowClient()

# Set version 1 as champion
client.set_registered_model_alias(MODEL_NAME, "champion", "1")
print("Version 1 → champion")

# Set version 2 as challenger
client.set_registered_model_alias(MODEL_NAME, "challenger", "2")
print("Version 2 → challenger")

# Now promote version 2 to champion
client.set_registered_model_alias(MODEL_NAME, "champion", "2")
print("Version 2 → promoted to champion")
