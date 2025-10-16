import mlflow
import mlflow.pyfunc
from mlfow_variables import load_model_name,load_server


# define the model saved's details/parameters
model_name=load_model_name()
model_stage="staging"

model_uri=f"models:/{model_name}/{model_stage}"


# set it:
mlflow.set_tracking_uri(load_server()) # or define a variable before that can be used 

# load the model -> mlflow.statsmodels still does the job here:
model=mlflow.pyfunc.load_model(model_uri=model_uri)

# save the model
import joblib
joblib.dump(model,"arma_model_registry_test.pkl")