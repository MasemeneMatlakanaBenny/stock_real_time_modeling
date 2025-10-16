from flask import Flask
import pandas as pd


app=Flask(__name__)
predictions=pd.read_csv("forecasts.csv")

@app.route('/forecasts')
def get_forecasts():
    return predictions

