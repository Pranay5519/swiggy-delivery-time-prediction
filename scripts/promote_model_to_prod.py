import mlflow
import os
import json
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# -----------------------
# Load ENV
# -----------------------
load_dotenv()

# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = "Pranay5519"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

TRACKING_URI = "https://dagshub.com/Pranay5519/swiggy-delivery-time-prediction.mlflow"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient(tracking_uri=TRACKING_URI)


# -----------------------
# Load model name
# -----------------------
def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


# -----------------------
# Promotion Logic
# -----------------------
def promote_staging_to_production(model_name: str):

    try:
        # archive current production
        try:
            old_prod = client.get_model_version_by_alias(model_name, "production")
            client.set_registered_model_alias(
                model_name,
                "archived",
                old_prod.version
            )
            print(f"Archived old production version: {old_prod.version}")

        except Exception:
            print("No existing production model")

        # get staging model
        staging_version = client.get_model_version_by_alias(
            model_name,
            "staging"
        ).version

        # promote
        client.set_registered_model_alias(
            model_name,
            "production",
            staging_version
        )

        # remove staging alias
        client.delete_registered_model_alias(
            model_name,
            "staging"
        )

        print(f"Promoted version {staging_version} → production")

    except Exception as e:
        print(f"Promotion failed: {e}")
        raise


# -----------------------
# Run
# -----------------------
model_name = load_model_information(
    "run_information.json"
)["model_name"]

promote_staging_to_production(model_name)