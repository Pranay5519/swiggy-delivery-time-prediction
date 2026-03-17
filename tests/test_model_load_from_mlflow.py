import mlflow
import pytest
import joblib
from mlflow.tracking import MlflowClient

# Tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Pranay5519/swiggy-delivery-time-prediction.mlflow"
)


@pytest.mark.parametrize(
    "model_name, alias",
    [
        ("delivery_time_pred_model_1", "production"),
    ],
)
def test_load_production_model_and_preprocessor(model_name, alias):

    client = MlflowClient()

    try:
   
        version_details = client.get_model_version_by_alias(
            model_name,
            alias,
        )

        model_uri = f"models:/{model_name}@{alias}"

     
        model = mlflow.sklearn.load_model(model_uri)

        preprocessor_path = mlflow.artifacts.download_artifacts(
            artifact_uri="mlflow-artifacts:/1cca07a4f4e44a08b502e757d798ade9/fa28573ea10a4567a68a3f8939df2033/artifacts/preprocessor.joblib"
        )

        preprocessor = joblib.load(preprocessor_path)

 
        assert model is not None
        assert preprocessor is not None
        assert hasattr(model, "predict")
        assert hasattr(preprocessor, "transform")

        print(
            f"Model version {version_details.version} "
            f"with alias '{alias}' loaded successfully."
        )

    except Exception as e:

        
        try:
            debug_source = client.get_model_version_by_alias(
                model_name,
                alias,
            ).source
            print(f"🔎 Debug Source Path → {debug_source}")
        except Exception:
            pass

        pytest.fail(
            f" Model / Preprocessor loading failed. "
            f"Check artifact logging path. Error → {e}"
        )