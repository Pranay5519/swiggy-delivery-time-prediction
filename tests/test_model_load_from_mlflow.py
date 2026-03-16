import mlflow
import joblib
from mlflow.tracking import MlflowClient

def load_model_and_preprocessor(model_name, model_alias, vectorizer_path):
    mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/swiggy-delivery-time-prediction.mlflow")


    client = MlflowClient()

    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


model , preprocessor = load_model_and_preprocessor(model_name="delivery_time_pred_model_1",
                                                   model_alias="production",
                                                   vectorizer_path="models\preprocessor.joblib")

print(type(model) ,type(preprocessor))

