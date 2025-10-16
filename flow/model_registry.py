import mlflow
from mlflow import MlflowClient
from mlfow_variables import load_exp_name,load_server,load_tags,load_model_name,load_client


# define the variables here again:
server=load_server()

# experiment name:
exp_name=load_exp_name()

# the tags name:
tags=load_tags()

# the model name:
model_name=load_model_name()

# now set the tracking uri within the mlflow
mlflow.set_tracking_uri(uri=server)

# now lets create the client:
client=load_client()

# now lets create the experiment:
stock_exp=client.create_experiment(
    name=exp_name,
    tags=tags
)

# now lets set the experiment within the mlflow:
mlflow.set_experiment(experiment_name=exp_name)

# lets reload the model:
import joblib

model=joblib.load("arima_model.pkl")


#lets register the model:

run_name="stock_run"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.statsmodels.log_model(statsmodels_model=model,
                                 registered_model_name=model_name)