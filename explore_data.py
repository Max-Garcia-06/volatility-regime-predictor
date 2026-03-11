import pandas as pd

df = pd.read_csv("spy_data.csv", index_col=0, parse_dates=True)
try:
    calls = pd.read_csv("spy_calls.csv")
    puts = pd.read_csv("spy_puts.csv") 
    current_price = df.iloc[-1]["Close"]
    
    calls['distance'] = abs(calls['strike'] - current_price)
    atm_call = calls.loc[calls['distance'].idxmin()]

    puts['distance'] = abs(puts['strike'] - current_price)
    atm_put = puts.loc[puts['distance'].idxmin()]

except Exception as e:
    print(f"Error exploring data: {e}")