import joblib
import pandas as pd
from datetime import datetime

test=pd.read_csv("test.csv")

# use the final model served in mlflow
model=joblib.load("arima_model.pkl")

forecasts=model.forecast(45)

n_forecasts = len(forecasts[20:])

last_time=pd.to_datetime(test['Datetime'].iloc[-1])

future_times = [
    last_time + pd.Timedelta(minutes=5 * (i + 1))
    for i in range(n_forecasts)
]

# the next forecasts:
forecast_df = pd.DataFrame({
    "Datetime": future_times,
    "Open": forecasts[20:]
})

forecast_df.to_csv("forecasts.csv")
