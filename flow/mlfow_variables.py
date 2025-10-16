server= "http://127.0.0.1:5000"

exp_name="Open Stock prediction"

exp_description="Predicting the stock prediction with ARMA model"

tags={
    "project_name":"stock_real_time_prediction",
    "project_date":"2025-10-11",
    "project_team":"Machine Learning and Data Science",
    "mlflow.note.content":exp_description

}

model_name="arma_model"


# loads the local host
def load_server():
    return server

# loads the experimentn name
def load_exp_name():
    return exp_name

# load the tags for the project
def load_tags():
    return tags

# load the model name
def load_model_name():
    return model_name

# load the client:
def load_client():
    from mlflow import MlflowClient
    client=MlflowClient(tracking_uri=server)
    
    return client

