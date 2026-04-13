import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("WALK-FORWARD VALIDATION")
print("=" * 60)

df = pd.read_csv('features.csv', index_col=0, parse_dates=True)

feature_cols = [
    'spy_return_1d', 'spy_return_5d', 'spy_return_10d', 'spy_return_20d',
    'rsi_14', 'bb_position', 'price_vs_ma50', 'price_vs_ma200',
    'vix_level', 'vix_return_1d', 'vix_ma_ratio', 'vix_percentile', 'vix_spike',
    'vix_term_structure', 'vol_risk_premium',
    'realized_vol_10d', 'realized_vol_20d', 'volume_ratio'
]

X     = df[feature_cols].values
y     = df['target'].values
dates = df.index

TRAIN_WINDOW = 400
TEST_WINDOW  = 60
STEP         = 60

results = []
fold    = 1
start   = 0

while start + TRAIN_WINDOW + TEST_WINDOW <= len(X):
    train_end = start + TRAIN_WINDOW
    test_end  = train_end + TEST_WINDOW

    X_train = X[start:train_end]
    y_train = y[start:train_end]
    X_test  = X[train_end:test_end]
    y_test  = y[train_end:test_end]

    # Scale
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_s, y_train)
    base_acc = accuracy_score(y_test, dummy.predict(X_test_s))

    # Logistic regression
    lr = LogisticRegression(C=0.1, class_weight='balanced',
                             max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]
    lr_pred  = lr.predict(X_test_s)
    lr_acc   = accuracy_score(y_test, lr_pred)
    lr_auc   = roc_auc_score(y_test, lr_proba) if len(np.unique(y_test)) > 1 else 0.5

    # XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=10,
        gamma=0.3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=scale,
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train_s, y_train)
    xgb_proba = xgb.predict_proba(X_test_s)[:, 1]
    xgb_pred  = xgb.predict(X_test_s)
    xgb_acc   = accuracy_score(y_test, xgb_pred)
    xgb_auc   = roc_auc_score(y_test, xgb_proba) if len(np.unique(y_test)) > 1 else 0.5

    test_start_date = dates[train_end].date()
    test_end_date   = dates[test_end - 1].date()

    results.append({
        'fold':           fold,
        'test_start':     test_start_date,
        'test_end':       test_end_date,
        'baseline':       base_acc,
        'lr_acc':         lr_acc,
        'lr_auc':         lr_auc,
        'lr_beats_base':  lr_acc > base_acc,
        'xgb_acc':        xgb_acc,
        'xgb_auc':        xgb_auc,
        'xgb_beats_base': xgb_acc > base_acc,
    })

    lr_flag  = "GOOD" if lr_acc  > base_acc else "BAD"
    xgb_flag = "GOOD" if xgb_acc > base_acc else "BAD"

    print(f"\n   Fold {fold}: {test_start_date} → {test_end_date}")
    print(f"   Baseline : {base_acc:.3f}")
    print(f"   LR       : Acc {lr_acc:.3f}  AUC {lr_auc:.3f}  {lr_flag}")
    print(f"   XGBoost  : Acc {xgb_acc:.3f}  AUC {xgb_auc:.3f}  {xgb_flag}")

    start += STEP
    fold  += 1

# Summarize
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("WALK-FORWARD SUMMARY")
print("=" * 60)

print(f"\n{'':30s} {'LR':>10} {'XGBoost':>10}")
print("-" * 52)
print(f"   {'Folds beat baseline':<27} "
      f"{results_df['lr_beats_base'].sum():>3} / {len(results_df)}"
      f"   {results_df['xgb_beats_base'].sum():>3} / {len(results_df)}")
print(f"   {'Avg Accuracy':<27} "
      f"{results_df['lr_acc'].mean():.4f}"
      f"     {results_df['xgb_acc'].mean():.4f}")
print(f"   {'Avg AUC':<27} "
      f"{results_df['lr_auc'].mean():.4f}"
      f"     {results_df['xgb_auc'].mean():.4f}")
print(f"   {'Avg Baseline':<27} "
      f"{results_df['baseline'].mean():.4f}")
print(f"   {'Avg Improvement (pp)':<27} "
      f"{(results_df['lr_acc']  - results_df['baseline']).mean()*100:+.2f}"
      f"        {(results_df['xgb_acc'] - results_df['baseline']).mean()*100:+.2f}")

# Visualizations
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
x = range(len(results_df))

# Accuracy plot
axes[0].plot(x, results_df['lr_acc'],
             marker='o', label='LR Accuracy',
             color='steelblue', linewidth=2)
axes[0].plot(x, results_df['xgb_acc'],
             marker='s', label='XGB Accuracy',
             color='darkorange', linewidth=2)
axes[0].plot(x, results_df['baseline'],
             marker='^', label='Baseline',
             color='red', linewidth=2, linestyle='--')
axes[0].set_title('Walk-Forward Accuracy by Fold — LR vs XGBoost',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(x)
axes[0].set_xticklabels([str(d) for d in results_df['test_start']],
                         rotation=45, ha='right')
axes[0].legend()
axes[0].grid(alpha=0.3)

# AUC plot
axes[1].plot(x, results_df['lr_auc'],
             marker='o', label='LR AUC',
             color='steelblue', linewidth=2)
axes[1].plot(x, results_df['xgb_auc'],
             marker='s', label='XGB AUC',
             color='darkorange', linewidth=2)
axes[1].axhline(y=0.5,  color='gray',  linestyle=':',  alpha=0.7, label='Random (0.50)')
axes[1].axhline(y=0.60, color='green', linestyle='--', alpha=0.7, label='Target (0.60)')
axes[1].set_title('Walk-Forward AUC by Fold — LR vs XGBoost',
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('ROC-AUC')
axes[1].set_xticks(x)
axes[1].set_xticklabels([str(d) for d in results_df['test_start']],
                         rotation=45, ha='right')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('walk_forward_results.png', dpi=300, bbox_inches='tight')
plt.close()

results_df.to_csv('walk_forward_results.csv', index=False)