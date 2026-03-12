import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, f1_score)
from sklearn.dummy import DummyClassifier
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("LOGISTIC REGRESSION MODEL TRAINING")

# Load
df = pd.read_csv('features.csv', index_col=0, parse_dates=True)
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Features
feature_cols = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d',
    'rsi_14', 'bb_position', 'price_vs_ma50', 'price_vs_ma200',
    'vix_level', 'vix_return_1d', 'vix_ma_ratio', 'vix_percentile', 'vix_spike',
    'realized_vol_10d', 'realized_vol_20d', 'volume_ratio'
]

missing = [col for col in feature_cols if col not in df.columns]
if missing:
    print(f"\nMissing columns: {missing}")
    exit()

X = df[feature_cols]
y = df['target']

print(f"\nTarget Distribution:")
print(f"   Positive (1): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"   Negative (0): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")

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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit ONLY on train
X_val_scaled   = scaler.transform(X_val)          # Transform val
X_test_scaled  = scaler.transform(X_test)         # Transform test

# Baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train_scaled, y_train)
baseline_acc = accuracy_score(y_test, dummy.predict(X_test_scaled))
print(f"\nTrue Baseline: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")

# Train
model = LogisticRegression(
    C=0.1,              
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

model.fit(X_train_scaled, y_train)


# Predictions
y_train_pred  = model.predict(X_train_scaled)
y_val_pred    = model.predict(X_val_scaled)
y_test_pred   = model.predict(X_test_scaled)

y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_val_proba   = model.predict_proba(X_val_scaled)[:, 1]
y_test_proba  = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val,   y_val_pred)
test_acc  = accuracy_score(y_test,  y_test_pred)

train_auc = roc_auc_score(y_train, y_train_proba)
val_auc   = roc_auc_score(y_val,   y_val_proba)
test_auc  = roc_auc_score(y_test,  y_test_proba)

print("RESULTS")

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

overfitting_gap = train_auc - test_auc
print(f"\nOverfitting Check (Train AUC - Test AUC):")
print(f"   Gap: {overfitting_gap:.4f}  {'Good' if overfitting_gap < 0.1 else 'Overfitting'}")

print(f"\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, y_test_pred)
print(f"Confusion Matrix (Test Set):")
print(f"              Predicted")
print(f"              Neg    Pos")
print(f"            ---------------")
print(f"Actual Neg | {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"       Pos | {cm[1,0]:4d}   {cm[1,1]:4d}")

# Threshold analysis
print(f"\nThreshold Analysis:")
print(f"{'Threshold':>10} {'Accuracy':>8} {'F1':>4} {'Pos%':>10}")
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
coef_df = pd.DataFrame({
    'feature':     feature_cols,
    'coefficient': model.coef_[0],
    'abs_coef':    np.abs(model.coef_[0])
}).sort_values('abs_coef', ascending=False)

print(f"\n Feature Coefficients (Logistic Regression):")
for _, row in coef_df.iterrows():
    direction = "bullish" if row['coefficient'] > 0 else "bearish"
    print(f"   {row['feature']:25s} {row['coefficient']:+.4f}  {direction}")

# Visualizations
# Feature Coefficients
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient']]
plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.8)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title('Feature Coefficients - Logistic Regression', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient (+ = bullish signal, - = bearish signal)', fontsize=11)
plt.tight_layout()
plt.savefig('lr_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Logistic Regression (Test Set)', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('lr_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Performance chart
fig, ax = plt.subplots(figsize=(10, 6))
splits = ['Train', 'Val', 'Test']
accs   = [train_acc, val_acc, test_acc]
aucs   = [train_auc, val_auc, test_auc]
x      = np.arange(len(splits))
width  = 0.35
ax.bar(x - width/2, accs, width, label='Accuracy',  alpha=0.8, color='steelblue')
ax.bar(x + width/2, aucs, width, label='ROC-AUC',   alpha=0.8, color='darkorange')
ax.axhline(y=baseline_acc, color='red',  linestyle='--', alpha=0.7, label=f'Baseline ({baseline_acc:.2f})')
ax.axhline(y=0.5,          color='gray', linestyle=':',  alpha=0.5, label='Random (0.50)')
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_ylim([0.3, 0.9])
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance - Logistic Regression', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('lr_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# Save
joblib.dump(model,  'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')