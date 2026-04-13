from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("XGBOOST MODEL TRAINING")
print("=" * 60)

# Load
print("\n📂 Loading features.csv...")
df = pd.read_csv('features.csv', index_col=0, parse_dates=True)
print(f"Loaded {len(df)} samples")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Features
feature_cols = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d',
    'rsi_14', 'bb_position', 'price_vs_ma50', 'price_vs_ma200',
    'vix_level', 'vix_return_1d', 'vix_ma_ratio', 'vix_percentile', 'vix_spike',
    'vix_term_structure', 'vol_risk_premium',
    'realized_vol_10d', 'realized_vol_20d', 'volume_ratio'
]

missing = [col for col in feature_cols if col not in df.columns]
if missing:
    print(f"\nERROR: Missing columns: {missing}")
    print(f"   Available: {df.columns.tolist()}")
    exit()

X = df[feature_cols]
y = df['target']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target Distribution:")
print(f"   High Vol (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"   Low  Vol (0): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

# Split
train_size = int(0.7 * len(X))
val_size   = int(0.15 * len(X))

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_val   = X.iloc[train_size:train_size + val_size]
y_val   = y.iloc[train_size:train_size + val_size]
X_test  = X.iloc[train_size + val_size:]
y_test  = y.iloc[train_size + val_size:]

print(f"\nData Split:")
print(f"   Train: {len(X_train)} ({X_train.index[0].date()} to {X_train.index[-1].date()})")
print(f"   Val:   {len(X_val)} ({X_val.index[0].date()} to {X_val.index[-1].date()})")
print(f"   Test:  {len(X_test)} ({X_test.index[0].date()} to {X_test.index[-1].date()})")

# Scale features (fit only on train)
scaler      = StandardScaler()
X_train     = scaler.fit_transform(X_train)
X_val       = scaler.transform(X_val)
X_test      = scaler.transform(X_test)

# Baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"\nTrue Baseline (always predict majority): {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")

# Class imbalance
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale     = neg_count / pos_count
print(f"Class ratio (neg/pos): {scale:.2f}")

# Train
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,     # ← Reduce overfitting
    gamma=0.1,              # ← Reduce overfitting
    scale_pos_weight=scale, # ← Handle class imbalance
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Predictions
y_train_pred  = xgb_model.predict(X_train)
y_val_pred    = xgb_model.predict(X_val)
y_test_pred   = xgb_model.predict(X_test)

y_train_proba = xgb_model.predict_proba(X_train)[:, 1]
y_val_proba   = xgb_model.predict_proba(X_val)[:, 1]
y_test_proba  = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate
train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val,   y_val_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)

train_auc = roc_auc_score(y_train, y_train_proba)
val_auc   = roc_auc_score(y_val,   y_val_proba)
test_auc  = roc_auc_score(y_test,  y_test_proba)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nAccuracy:")
print(f"   Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"   Val:   {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"   Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")

print(f"\nROC-AUC:")
print(f"   Train: {train_auc:.4f}")
print(f"   Val:   {val_auc:.4f}")
print(f"   Test:  {test_auc:.4f}")

diff = test_acc - baseline_acc
flag = "Beats baseline" if diff > 0 else "Worse than baseline"
print(f"\nBaseline vs Model:")
print(f"   Baseline: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
print(f"   Model:    {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"   Gap:      {diff*100:+.2f} percentage points  {flag}")

overfit_gap = train_auc - test_auc
print(f"\nOverfitting Check (Train AUC - Test AUC):")
print(f"   Gap: {overfit_gap:.4f}  {'Good' if overfit_gap < 0.1 else 'Overfitting'}")

print(f"\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred,
                             target_names=['Low Vol', 'High Vol']))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"Confusion Matrix (Test Set):")
print(f"                 Predicted")
print(f"              Low    High")
print(f"           ---------------")
print(f"Actual Low  | {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"       High | {cm[1,0]:4d}   {cm[1,1]:4d}")

# Threshold analysis
print(f"\nThreshold Analysis:")
print(f"{'Threshold':>10} {'Accuracy':>8} {'F1':>4} {'HighVol%':>10}")
print("-" * 45)

best_f1, best_threshold = 0, 0.5
for threshold in np.arange(0.3, 0.75, 0.05):
    preds   = (y_test_proba >= threshold).astype(int)
    acc     = accuracy_score(y_test, preds)
    f1      = f1_score(y_test, preds, zero_division=0)
    pos_pct = preds.mean() * 100
    flag    = " ← current" if abs(threshold - 0.5) < 0.01 else ""
    print(f"   {threshold:.2f}    {acc:.4f}    {f1:.4f}    {pos_pct:.1f}%{flag}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest Threshold: {best_threshold:.2f}  (F1 = {best_f1:.4f})")

# Feature importance
importance_df = pd.DataFrame({
    'feature':    feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nFeature Importance:")
for _, row in importance_df.iterrows():
    bar = '█' * int(row['importance'] * 100)
    print(f"   {row['feature']:25s} {row['importance']:.4f}  {bar}")

importance_df.to_csv('xgb_feature_importance.csv', index=False)

# Visualizations
# Feature Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature',
            hue='feature', palette='viridis', legend=False)
plt.title('Feature Importance - XGBoost', fontsize=14, fontweight='bold')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('xgb_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Vol', 'High Vol'],
            yticklabels=['Low Vol', 'High Vol'])
plt.title('Confusion Matrix - XGBoost (Test Set)', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Performance Chart
fig, ax = plt.subplots(figsize=(10, 6))
splits = ['Train', 'Val', 'Test']
accs   = [train_acc, val_acc, test_acc]
aucs   = [train_auc, val_auc, test_auc]
x      = np.arange(len(splits))
width  = 0.35
ax.bar(x - width/2, accs, width, label='Accuracy',  alpha=0.8, color='steelblue')
ax.bar(x + width/2, aucs, width, label='ROC-AUC',   alpha=0.8, color='darkorange')
ax.axhline(y=baseline_acc, color='red',  linestyle='--',
           alpha=0.7, label=f'Baseline ({baseline_acc:.2f})')
ax.axhline(y=0.5,          color='gray', linestyle=':',
           alpha=0.5, label='Random (0.50)')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylim([0.3, 0.95])
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance - XGBoost', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('xgb_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# Calibrate probabilities using validation set (Platt scaling)
from sklearn.linear_model import LogisticRegression as PlattScaler
xgb_calibrator = PlattScaler(C=1.0)
xgb_calibrator.fit(y_val_proba.reshape(-1, 1), y_val)
print("\nProbability calibration applied (Platt scaling on val set)")

# Save
joblib.dump(xgb_model,      'xgboost_model.pkl')
joblib.dump(xgb_calibrator, 'xgb_calibrator.pkl')
