"""Live daily volatility regime prediction."""

from datetime import datetime, timedelta
import warnings

import joblib
import pandas as pd

from features import (
    download_market_data,
    drop_incomplete_rows,
    engineer_features,
    latest_feature_row,
)

warnings.filterwarnings("ignore")

print("TODAY'S VOLATILITY REGIME PREDICTION")
print(f"\nRun time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Need ~250 trading days for MA200 and rolling windows
end = datetime.today()
start = end - timedelta(days=400)

spy, vix, vix9d, volume = download_market_data(start=start, end=end)
df = engineer_features(spy, vix, vix9d, volume, include_target=False)
df = drop_incomplete_rows(df, for_training=False)

X_today, pred_date = latest_feature_row(df)
row = df.loc[pred_date]

lr = joblib.load("logistic_regression_model.pkl")
lr_cal = joblib.load("lr_calibrator.pkl")
xgb = joblib.load("xgboost_model.pkl")
xgb_cal = joblib.load("xgb_calibrator.pkl")
scaler = joblib.load("scaler.pkl")

X_scaled = scaler.transform(X_today)

lr_raw = lr.predict_proba(X_scaled)[0][1]
xgb_raw = xgb.predict_proba(X_scaled)[0][1]
lr_proba = float(lr_cal.predict_proba([[lr_raw]])[0][1])
xgb_proba = float(xgb_cal.predict_proba([[xgb_raw]])[0][1])
ens_proba = (lr_proba + xgb_proba) / 2

print(f"\nPrediction for: {pred_date.date()}")
print(f"   SPY Close:   ${row['spy_close']:.2f}")
print(f"   VIX Level:   {row['vix_close']:.2f}")
print(f"   Realized Vol (20d): {row['realized_vol_20d']:.1%}")

print("\nModel Probabilities (High Vol):")
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
    print("\nOptions Implication:")
    print("   → Consider LONG options (straddles, strangles)")
    print("   → Elevated IV expected — buy vol before it spikes")
else:
    print("\nOptions Implication:")
    print("   → Consider SHORT options (iron condors, covered calls)")
    print("   → Low vol expected — collect theta decay")

log = pd.DataFrame(
    [
        {
            "date": pred_date.date(),
            "spy_close": round(row["spy_close"], 2),
            "vix_level": round(row["vix_close"], 2),
            "lr_proba": round(lr_proba, 4),
            "xgb_proba": round(xgb_proba, 4),
            "ens_proba": round(ens_proba, 4),
            "regime": "HIGH" if ens_proba >= 0.5 else "LOW",
            "signal_strength": signal.split()[0],
        }
    ]
)

try:
    existing = pd.read_csv("prediction_log.csv")
    existing = existing[existing["date"] != str(pred_date.date())]
    existing = existing.drop(columns=["confidence"], errors="ignore")
    log = pd.concat([existing, log], ignore_index=True)
except FileNotFoundError:
    pass

log.to_csv("prediction_log.csv", index=False)
