import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

spy = yf.Ticker("SPY")
df = spy.history(period="2y")
df.to_csv("spy_data.csv")

try:
    expirations = spy.options
    opt_chain = spy.option_chain(expirations[0])
    calls = opt_chain.calls
    puts = opt_chain.puts
    calls.to_csv("spy_calls.csv")
    puts.to_csv("spy_puts.csv")

except Exception as e:
    print(f"Error fetching options data: {e}")