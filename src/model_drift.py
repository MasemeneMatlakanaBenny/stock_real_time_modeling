import pandas as pd
import numpy as np

import warnings

np.NInf=np.inf
np.Inf=np.inf

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import LabelDrift

import joblib


# load the test ,train and forecasts
train=pd.read_csv("train.csv").drop("Datetime",axis=1)
test=pd.read_csv("test.csv").drop("Datetime",axis=1)
forecasts=pd.read("forecasts.csv").drop("Datetime",axis=1)
# reload the model
model=joblib.load("arima-model.pkl")

# define the dataset in a deepchecks format
test_data=Dataset(train.drop("Datetime",axis=1),label="Open",cat_features=[])
train_data=Dataset(test.drop("Datetime",axis=1),label="Open",cat_features=[])

pred_data=Dataset(forecasts.drop("Datetime",axis=1),label="Open",cat_features=[])


check=LabelDrift()

results=check.run(train,forecasts)

print(results)