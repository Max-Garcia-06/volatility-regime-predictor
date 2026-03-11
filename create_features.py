import yfinance as yf
import pandas as pd
import numpy as np

print("=" * 60)
print("CREATING FEATURES.CSV (IMPROVED)")
print("=" * 60)

# Download data
print("\nDownloading data...")
spy = yf.download('SPY', period='5y', progress=False)  # ← More data!
vix = yf.download('^VIX', period='5y', progress=False)

spy.columns = spy.columns.get_level_values(0)
vix.columns = vix.columns.get_level_values(0)

print(f"Downloaded {len(spy)} days of data")

df = pd.DataFrame({
    'spy_close': spy['Close'],
    'volume':    spy['Volume'],
    'vix_close': vix['Close']
})

# Predict if next 10 days will have HIGH volatility
df['future_vol_10d'] = (
    df['spy_close'].pct_change()
    .rolling(10).std()
    .shift(-10) * np.sqrt(252)
)
# Target: will next 10 days be HIGH volatility?
df['target'] = (
    df['future_vol_10d'] > df['future_vol_10d'].rolling(60).median()
).astype(int)

print("\n🔧 Engineering features...")

# Price momentum
df['spy_return_1d'] = df['spy_close'].pct_change(1).shift(1)
df['spy_return_5d'] = df['spy_close'].pct_change(5).shift(1)
df['spy_return_10d'] = df['spy_close'].pct_change(10).shift(1)
df['spy_return_20d'] = df['spy_close'].pct_change(20).shift(1)

# RSI
delta = df['spy_close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df['rsi_14'] = (100 - (100 / (1 + gain / loss))).shift(1)

# Bollinger bands
ma20 = df['spy_close'].rolling(20).mean()
std20 = df['spy_close'].rolling(20).std()
df['bb_position'] = ((df['spy_close'] - ma20) / (2 * std20)).shift(1)

# Price vs ma
df['price_vs_ma50']  = (df['spy_close'] / df['spy_close'].rolling(50).mean() - 1).shift(1)
df['price_vs_ma200'] = (df['spy_close'] / df['spy_close'].rolling(200).mean() - 1).shift(1)

# VIX
df['vix_level']     = df['vix_close'].shift(1)
df['vix_return_1d'] = df['vix_close'].pct_change(1).shift(1)
df['vix_ma_10']     = df['vix_close'].rolling(10).mean().shift(1)
df['vix_ma_20']     = df['vix_close'].rolling(20).mean().shift(1)
df['vix_ma_ratio']  = df['vix_ma_10'] / df['vix_ma_20']

# VIX percentile
df['vix_percentile'] = df['vix_close'].rolling(60).rank(pct=True).shift(1)

# VIX spike detection
df['vix_spike'] = (df['vix_close'] > df['vix_close'].rolling(20).mean() * 1.2).astype(int).shift(1)

# Vol
df['realized_vol_10d'] = (df['spy_close'].pct_change().rolling(10).std() * np.sqrt(252)).shift(1)
df['realized_vol_20d'] = (df['spy_close'].pct_change().rolling(20).std() * np.sqrt(252)).shift(1)

# ── Volume ───────────────────────────────
df['volume_ratio'] = (df['volume'] / df['volume'].rolling(20).mean()).shift(1)

# Clean up
print(f"\n🧹 Rows before dropna: {len(df)}")
df = df.dropna()
print(f"Rows after dropna:  {len(df)}")

print(f"\nTarget Distribution:")
print(f"   Positive (1): {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
print(f"   Negative (0): {(1-df['target']).sum()} ({(1-df['target'].mean())*100:.1f}%)")

# Check correlations
feature_cols = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d',
    'rsi_14', 'bb_position', 'price_vs_ma50', 'price_vs_ma200',
    'vix_level', 'vix_return_1d', 'vix_ma_ratio', 'vix_percentile', 'vix_spike',
    'realized_vol_10d', 'realized_vol_20d', 'volume_ratio'
]

print(f"\n📊 Feature-Target Correlations:")
correlations = df[feature_cols].corrwith(df['target']).sort_values(ascending=False)
for feat, corr in correlations.items():
    flag = "SIGNAL" if abs(corr) > 0.1 else ("✅" if abs(corr) > 0.05 else "⚠️  weak")
    print(f"   {flag} {feat:25s}  {corr:.4f}")

df.to_csv('features.csv')
