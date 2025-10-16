import mlflow
import mlflow.pyfunc
from mlfow_variables import load_model_name


# define the model saved's details/parameters
model_name=load_model_name()
model_version="1"

model_uri=f"models:/{model_name}/{model_version}"


# load the model -> mlflow.statsmodels still does the job here:
model=mlflow.pyfunc.load_model(model_uri=model_uri)

# save the model
import joblib
joblib.dump(model,"arma_model_registry_test.pkl")