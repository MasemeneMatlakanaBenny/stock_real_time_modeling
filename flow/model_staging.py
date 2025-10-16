import mlflow
from mlflow import MlflowClient
from mlfow_variables import load_model_name,load_server,load_client


# define the local host and model name:
server=load_server()
model_name=load_model_name()
model_version="1"
 
# set it into the mlflow
mlflow.set_tracking_uri(server)


# define the client
client=load_client()

# now stage the model:
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="staging"
)

