from prefect import task,flow


@task(retries=3,retry_delay_seconds=30,timeout_seconds=180)
def read_data(path:str):

    """
    Task for loading the dataset
     From the src/data_processing.py ,we loaded the dataset and saved it 
     so here we reload it by reading it from csv file/
    
    """
    # import libraries -> pandas in this case ,just pandas only.
    import pandas as pd

    return pd.read_csv(path)


@task(retries=3,retry_delay_seconds=30,timeout_seconds=60)
def load_train_test(train_path,test_path):
    """
    Task for loading the dataset
     From the src/data_split.py ,we saved both train and test files
     Hence here we reload both the files by reading it from csv file/
    
    """
    #import libs-> just pandas only
    import pandas as pd

    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    
    return train,test

@task 
def load_model(model_path):
    """
    Load the final model that has been moved to production in mlflow
    """
    # import joblib only
    import joblib
    import mlflow 
    import mlflow.pyfunc
    from flow.mlfow_variables import load_model_name,load_client,load_server

    mlflow.set_tracking_uri(uri=load_server())

    # variables that will be used to define the model uri
    model_name=load_model_name()
    model_stage="production"

    # get the model uri:
    model_uri=f"models:/{model_name}/{model_stage}"

    # load the model -> mlflow.statsmodels still does the job here:
    model=mlflow.pyfunc.load_model(model_uri=model_uri)


    return model

@task   
def save_model(model):
    # save the model
    import joblib
    joblib.dump(model,"arma_prod_model.pkl")

@task
def get_forecasts(model):
    model.forecast(10)