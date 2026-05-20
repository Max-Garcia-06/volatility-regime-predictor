"""Download market data and write features.csv for model training."""

import pandas as pd

from features import (
    FEATURE_COLS,
    download_market_data,
    drop_incomplete_rows,
    engineer_features,
)

print("=" * 60)
print("CREATING FEATURES.CSV")
print("=" * 60)

print("\nDownloading data...")
spy, vix, vix9d, volume = download_market_data(period="5y")
print(f"Downloaded {len(spy)} days of SPY data")

print("\nEngineering features...")
df = engineer_features(spy, vix, vix9d, volume, include_target=True)
print(f"Rows before dropna: {len(df)}")
df = drop_incomplete_rows(df, for_training=True)
print(f"Rows after dropna:  {len(df)}")

print("\nTarget distribution:")
print(f"   Positive (1): {df['target'].sum()} ({df['target'].mean() * 100:.1f}%)")
print(f"   Negative (0): {(1 - df['target']).sum()} ({(1 - df['target'].mean()) * 100:.1f}%)")

print("\nFeature-target correlations:")
correlations = df[FEATURE_COLS].corrwith(df["target"]).sort_values(ascending=False)
for feat, corr in correlations.items():
    flag = "SIGNAL" if abs(corr) > 0.1 else ("ok" if abs(corr) > 0.05 else "weak")
    print(f"   {flag:6s} {feat:25s}  {corr:.4f}")

df.to_csv("features.csv")
print("\nSaved features.csv")
