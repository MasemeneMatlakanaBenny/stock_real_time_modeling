import mlflow
from mlfow_variables import load_model_name,load_server,load_client

server=load_server()

model_name=load_model_name()

client=load_client()

# now lets move the model to production:

mlflow.set_tracking_uri(uri=server)

# now lets transition the model:

client.transition_model_version_stage(
    name=model_name,
    model_version="1",
    stage="production"
)

