import pandas as pd

# reload the dataset extracted from the yfinance lib
dataset=pd.read_csv("nvidia_stock_data.csv")

# without shuffling since it is a time series data:
# N -> length of the dataset

N = len(dataset)

# train length
train_len = N - 20

# test length
test_len = 20

# now get the train and test sets
train = dataset[:train_len]
test = dataset[train_len:train_len + test_len]
   
# save the train and test sets:
train[["Datetime","Open"]].to_csv("train.csv")
test[["Datetime","Open"]].to_csv("test.csv")


