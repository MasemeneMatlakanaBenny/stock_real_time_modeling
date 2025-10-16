# This is a full prefect ml pipeline that does everything:
from prefect import task,flow
from prefect.server.schemas.schedules import IntervalSchedule

@task(retries=3,retry_delay_seconds=30,timeout_seconds=240)
def load_data():
    """
    A function that loads the raw data
    """
    import yfinance as yf
    ticker = "NVDA"
    nvda = yf.Ticker(ticker)
    period=30
    data = nvda.history(period=f"{period}d", interval="15m") 
    data=data.reset_index()
    data['Datetime']= data["Datetime"].dt.tz_convert("Africa/Johannesburg").dt.strftime("%Y-%m-%d %H:%M")
    data.to_csv("nvidia_stock_data.csv")

@task
def data_split(path:str):
    import pandas as pd
    dataset=pd.read_csv(path)

    # without shuffling since it is a time series data:
    # # N -> length of the dataset
    N = len(dataset)

    # train length
    train_len = N - 20
    # test length
    test_len = 20


    # now get the train and test sets
    train = dataset[:train_len]
    test = dataset[train_len:train_len + test_len]
    # save the train and test sets:
    train_df=train[["Datetime","Open"]]
    test_df=test[["Datetime","Open"]]
    train_df.to_csv("train.csv")
    test_df.to_csv("test.csv")

    return train_df,test_df



@task
def model_training(train_path):
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    # reload the train set now
    train=pd.read_csv(train_path)
    # define an ARMA model-> p=1 and q=1 
    model=ARIMA(train['Open'],order=(1,0,1))
    # fit the model
    model=model.fit()
    

    return model


@task   
def save_model(model):
    # save the model
    import joblib
    joblib.dump(model,"arma_prod_model.pkl")

@task
def create_experiment():
    import mlflow
    from flow.mlfow_variables import load_server,load_exp_name,load_tags,load_client

    # define the variables first:
    server=load_server()
    exp_name=load_exp_name()
    tags=load_tags()

    # now set the mlflow :
    mlflow.set_tracking_uri(uri=server)

    # create the client:
    client=load_client()

    # now use the client to create the experiment:
    stock_exp=client.create_experiment(
        name=exp_name,
        tags=tags
    )

@task
def model_registry(model_fitted):
    import mlflow
    from flow.mlfow_variables import load_exp_name,load_model_name,load_server

    # define the variables first:
    server=load_server()
    exp_name=load_exp_name()
    model_name=load_model_name()

    # create the model variable -> optional:
    model=model_fitted

    # set the experiment uri in the mlflow:
    mlflow.set_tracking_uri(server)

    # set the experiment name in the mlflow too:
    mlflow.set_experiment(experiment_name=exp_name)

    # register the model:
    with mlflow.start_run(run_name="run_stock_pipeline") as run:
        mlflow.statsmodels.log_model(statsmodels_model=model,registered_model_name=model_name)

@task
def test_model_registry():
    import mlflow
    import mlflow.pyfunc
    from flow.mlfow_variables import load_model_name,load_server
    
    # define certain variables -> model_name and version
    model_name=load_model_name()
    model_version="1"
    server=load_server()

    # set the tracking uri in mlflow
    mlflow.set_tracking_uri(uri=server)

    # now test the model registry:
    ## defint the model uri
    model_uri=f"models:/{model_name}/{model_version}"

    model=mlflow.pyfunc.load_model(model_uri=model_uri)

    # import joblib -> save the model
    import joblib

    joblib.dump(model,"model_registry_prefect_pipeline.pkl")


@task
def model_staging():
    import mlflow
    from mlflow import MlflowClient
    from flow.mlfow_variables import load_model_name,load_server,load_client

    # define the local host and model name:
    server=load_server()
    model_name=load_model_name()
    model_version="1"

    # set it into the mlflow
    mlflow.set_tracking_uri(server)

    #define the client
    client=load_client()

    #now stage the model:
    client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="staging")

@task
def test_model_staging():
    import mlflow
    import mlflow.pyfunc
    from flow.mlfow_variables import load_model_name,load_server
    # define the model saved's details/parameters
    model_name=load_model_name()
    model_stage="staging"
    model_uri=f"models:/{model_name}/{model_stage}"

    # set the tracking uri
    mlflow.set_tracking_uri(load_server()) # or define a variable before that can be used 
    
    # load the model -> mlflow.statsmodels still does the job here:
    model=mlflow.pyfunc.load_model(model_uri=model_uri) 
    # save the model
    import joblib
    joblib.dump(model,"model_registry_test_prefect_pipeline.pkl")

@task
def model_production():
    import mlflow
    from flow.mlfow_variables import load_model_name,load_server,load_client
    
    # define certain variables first:
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

@task
def test_model_production():
    import mlflow
    import mlflow.pyfunc
    from flow.mlfow_variables import load_model_name,load_server
    
    # define certain model params:
    server=load_server()
    model_name=load_model_name()
    model_stage="production"

    # get the model uri:
    model_uri=f"models:/{model_name}/{model_stage}"
    
    # now load the model:
    model=mlflow.pyfunc.load_model(model_uri=model_uri)
    
    # save the model:
    import joblib
    joblib.dump(model,"arma_model_final.pkl")


@flow
def model_pipeline_workflow():
    dataset=load_data()
    train,test=data_split(path="nvidia_stock_data.csv")
    model=model_training("train.csv")
    save_model(model=model)
    create_experiment()
    model_registry(model_fitted=model)
    test_model_registry()
    model_staging()
    test_model_staging()
    model_production()
    test_model_production()


if __name__=="__main__":
    from datetime import datetime,timedelta
    
  # schedule the workflow:
    schedule_instances=IntervalSchedule(
        interval=timedelta(hours=5),
        anchor_date=datetime(2025,10,13),
        timezone="Africa/Johannesburg"
    )

  # deploy:
    model_pipeline_workflow.serve(
        name="stock_automated_workflow",
        schedule=schedule_instances.dict()
    )
