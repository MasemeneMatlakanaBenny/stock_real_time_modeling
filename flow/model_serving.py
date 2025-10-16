import mlflow
import mlflow.pyfunc
from mlfow_variables import load_model_name,load_server

server=load_server()

model_name=load_model_name()
model_stage="production"

model_uri=f"models:/{model_name}/{model_stage}"

# now load the model:

model=mlflow.pyfunc.load_model(model_uri=model_uri)

# save the model:
import joblib

joblib.dump(model,"arma_model_final.pkl")