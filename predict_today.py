import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("TODAY'S VOLATILITY REGIME PREDICTION")
print(f"\nRun time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Download data
end   = datetime.today()
start = end - timedelta(days=365)  # 1 year to compute features

spy_raw  = yf.download('SPY',    start=start, end=end, auto_adjust=True, progress=False)
vix_raw  = yf.download('^VIX',   start=start, end=end, auto_adjust=True, progress=False)
vix9d_raw = yf.download('^VIX9D', start=start, end=end, auto_adjust=True, progress=False)

spy    = spy_raw['Close'].squeeze()
volume = spy_raw['Volume'].squeeze()
vix    = vix_raw['Close'].squeeze()
vix9d  = vix9d_raw['Close'].squeeze()

# Engineer features
df = pd.DataFrame(index=spy.index)
df['spy_close'] = spy
df['vix_close']  = vix

df['spy_return_1d']  = spy.pct_change(1)
df['spy_return_5d']  = spy.pct_change(5)
df['spy_return_10d'] = spy.pct_change(10)
df['spy_return_20d'] = spy.pct_change(20)

delta = spy.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = -delta.clip(upper=0).rolling(14).mean()
df['rsi_14'] = 100 - (100 / (1 + gain / loss))

ma20  = spy.rolling(20).mean()
std20 = spy.rolling(20).std()
df['bb_position'] = (spy - ma20) / (2 * std20)

df['price_vs_ma50']  = spy / spy.rolling(50).mean() - 1
df['price_vs_ma200'] = spy / spy.rolling(200).mean() - 1

df['vix_level']      = vix
df['vix_return_1d']  = vix.pct_change(1)
vix_ma10             = vix.rolling(10).mean()
vix_ma20             = vix.rolling(20).mean()
df['vix_ma_ratio']   = vix_ma10 / vix_ma20
df['vix_percentile'] = vix.rolling(60).rank(pct=True)
df['vix_spike']          = (vix > vix_ma20 * 1.2).astype(int).shift(1)
df['vix_term_structure'] = vix9d.reindex(df.index) / vix

realized_vol_20d_raw     = spy.pct_change().rolling(20).std() * np.sqrt(252)
df['vol_risk_premium']   = vix / 100 - realized_vol_20d_raw

df['realized_vol_10d'] = spy.pct_change().rolling(10).std() * np.sqrt(252)
df['realized_vol_20d'] = realized_vol_20d_raw

df['volume_ratio'] = volume / volume.rolling(20).mean()

df.dropna(inplace=True)

feature_cols = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d',
    'rsi_14', 'bb_position', 'price_vs_ma50', 'price_vs_ma200',
    'vix_level', 'vix_return_1d', 'vix_ma_ratio', 'vix_percentile', 'vix_spike',
    'vix_term_structure', 'vol_risk_premium',
    'realized_vol_10d', 'realized_vol_20d', 'volume_ratio'
]

# Load models and calibrators
lr      = joblib.load('logistic_regression_model.pkl')
lr_cal  = joblib.load('lr_calibrator.pkl')
xgb     = joblib.load('xgboost_model.pkl')
xgb_cal = joblib.load('xgb_calibrator.pkl')
scaler  = joblib.load('scaler.pkl')

X_today  = df[feature_cols].iloc[[-1]]
X_scaled = scaler.transform(X_today)

lr_raw    = lr.predict_proba(X_scaled)[0][1]
xgb_raw   = xgb.predict_proba(X_scaled)[0][1]
lr_proba  = float(lr_cal.predict_proba([[lr_raw]])[0][1])
xgb_proba = float(xgb_cal.predict_proba([[xgb_raw]])[0][1])
ens_proba = (lr_proba + xgb_proba) / 2

# Output
date = df.index[-1].date()

print(f"\nPrediction for: {date}")
print(f"   SPY Close:   ${df['spy_close'].iloc[-1]:.2f}")
print(f"   VIX Level:   {df['vix_close'].iloc[-1]:.2f}")
print(f"   Realized Vol (20d): {df['realized_vol_20d'].iloc[-1]:.1%}")

print(f"\nModel Probabilities (High Vol):")
print(f"   Logistic Regression: {lr_proba:.3f}")
print(f"   XGBoost:             {xgb_proba:.3f}")
print(f"   Ensemble:            {ens_proba:.3f}")

regime = "HIGH VOLATILITY" if ens_proba >= 0.5 else "LOW VOLATILITY"

dist = abs(ens_proba - 0.5)
if dist >= 0.20:
    signal = "Strong"
elif dist >= 0.10:
    signal = "Moderate"
else:
    signal = "Weak  (near coin-flip, consider sitting out)"

print(f"\nPredicted Regime:   {regime}")
print(f"   High-vol probability: {ens_proba:.1%}  [{signal}]")

if ens_proba >= 0.5:
    print(f"\nOptions Implication:")
    print(f"   → Consider LONG options (straddles, strangles)")
    print(f"   → Elevated IV expected — buy vol before it spikes")
else:
    print(f"\nOptions Implication:")
    print(f"   → Consider SHORT options (iron condors, covered calls)")
    print(f"   → Low vol expected — collect theta decay")

# Log prediction
log = pd.DataFrame([{
    'date':        date,
    'spy_close':   round(df['spy_close'].iloc[-1], 2),
    'vix_level':   round(df['vix_close'].iloc[-1], 2),
    'lr_proba':    round(lr_proba, 4),
    'xgb_proba':   round(xgb_proba, 4),
    'ens_proba':      round(ens_proba, 4),
    'regime':         'HIGH' if ens_proba >= 0.5 else 'LOW',
    'signal_strength': signal.split()[0]
}])

try:
    existing = pd.read_csv('prediction_log.csv')
    existing = existing[existing['date'] != str(date)]
    existing = existing.drop(columns=['confidence'], errors='ignore')
    log = pd.concat([existing, log], ignore_index=True)
except FileNotFoundError:
    pass

log.to_csv('prediction_log.csv', index=False)
