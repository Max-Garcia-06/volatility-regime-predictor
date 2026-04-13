import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('features.csv', index_col=0, parse_dates=True)

feature_cols = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d',
    'rsi_14', 'bb_position', 'price_vs_ma50', 'price_vs_ma200',
    'vix_level', 'vix_return_1d', 'vix_ma_ratio', 'vix_percentile', 'vix_spike',
    'vix_term_structure', 'vol_risk_premium',
    'realized_vol_10d', 'realized_vol_20d', 'volume_ratio'
]

X = df[feature_cols]
y = df['target']

train_size = int(0.7 * len(X))
val_size   = int(0.15 * len(X))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_val   = X.iloc[train_size:train_size + val_size]
y_val   = y.iloc[train_size:train_size + val_size]
X_test  = X.iloc[train_size + val_size:]
y_test  = y.iloc[train_size + val_size:]

model = joblib.load('random_forest_model.pkl')

# Get probabilities
y_test_proba = model.predict_proba(X_test)[:, 1]

print("=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)
print(f"\n{'Threshold':>10} {'Accuracy':>10} {'F1':>10} {'Pos%':>10}")
print("-" * 45)

best_f1 = 0
best_threshold = 0.5

for threshold in np.arange(0.3, 0.8, 0.05):
    preds = (y_test_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, zero_division=0)
    pos_pct = preds.mean() * 100
    flag = " ← current" if abs(threshold - 0.5) < 0.01 else ""
    print(f"   {threshold:.2f}    {acc:.4f}    {f1:.4f}    {pos_pct:.1f}%{flag}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest Threshold: {best_threshold:.2f}  (F1 = {best_f1:.4f})")
print(f"\nTest AUC: {roc_auc_score(y_test, y_test_proba):.4f}")