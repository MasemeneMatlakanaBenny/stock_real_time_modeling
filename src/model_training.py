import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# reload the train set now
train=pd.read_csv("train.csv")

# define an ARMA model-> p=1 and q=1 
model=ARIMA(train['Open'],order=(1,0,1))

# fit the model
model=model.fit()

# get the model summary
print(model.summary())

# save the model now:
import joblib
joblib.dump(model,"arima_model.pkl")