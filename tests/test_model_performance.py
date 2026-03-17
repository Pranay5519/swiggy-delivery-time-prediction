import pytest
import mlflow
import dagshub
import json
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from mlflow.tracking import MlflowClient
dagshub.init(repo_owner='Pranay5519', 
             repo_name='swiggy-delivery-time-prediction', 
             mlflow=True)
# authoize
# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/swiggy-delivery-time-prediction.mlflow")


def load_model_and_preprocessor(model_name, model_alias):
    mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/swiggy-delivery-time-prediction.mlflow")


    client = MlflowClient()

    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    preprocessor_path = mlflow.artifacts.download_artifacts(
        artifact_uri="mlflow-artifacts:/1cca07a4f4e44a08b502e757d798ade9/fa28573ea10a4567a68a3f8939df2033/artifacts/preprocessor.joblib"
    )

    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor

model_name="delivery_time_pred_model_1"

model , preprocessor = load_model_and_preprocessor(model_name="delivery_time_pred_model_1",
                                                   model_alias="production" )



# build the model pipeline
model_pipe = Pipeline(steps=[
    ('preprocess',preprocessor),
    ("regressor",model)
])
root_path = Path(__file__).parent.parent

test_data_path = root_path / "data" / "interim" / "test.csv"

@pytest.mark.parametrize(argnames="model_pipe, test_data_path, threshold_error",
                        argvalues=[(model_pipe, test_data_path, 5)])
def test_model_performance(model_pipe,test_data_path,threshold_error):
    # load test data
    df = pd.read_csv(test_data_path)
    
    # drop the missing values
    df.dropna(inplace=True)
    
    # make X and y
    X = df.drop(columns=["time_taken"])
    y = df['time_taken']
    
    # get the predictions
    y_pred = model_pipe.predict(X)
    
    # calculate the mean error
    mean_error = mean_absolute_error(y,y_pred)
    print(mean_error)
    # check for performance
    assert mean_error <= threshold_error, f"The model does not pass the performance threshold of {threshold_error} minutes"
    print("The avg error is", mean_error)
    
    print(f"The {model_name} model passed the performance test")
    
    
    
# pytest -s .\tests\test_model_preformance.py --> to print all the print statements