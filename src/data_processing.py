import yfinance as yf

ticker = "NVDA"

nvda = yf.Ticker(ticker)

period=30

data = nvda.history(period=f"{period}d", interval="15m") 

data=data.reset_index()

data['Datetime']= data["Datetime"].dt.tz_convert("Africa/Johannesburg").dt.strftime("%Y-%m-%d %H:%M")

data.to_csv("nvidia_stock_data.csv")
  