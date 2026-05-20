import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import warnings

from features import (
    FEATURE_COLS,
    RANDOM_STATE,
    WALK_FORWARD_STEP,
    WALK_FORWARD_TEST,
    WALK_FORWARD_TRAIN,
)

warnings.filterwarnings('ignore')

print("=" * 60)
print("ENSEMBLE MODEL (LR + XGBoost)")
print("=" * 60)

df = pd.read_csv('features.csv', index_col=0, parse_dates=True)

X     = df[FEATURE_COLS].values
y     = df['target'].values
dates = df.index

TRAIN_WINDOW = WALK_FORWARD_TRAIN
TEST_WINDOW  = WALK_FORWARD_TEST
STEP         = WALK_FORWARD_STEP

results = []
fold    = 1
start   = 0

print(f"\n{'Fold':<6} {'Period':<30} {'Base':>6} {'LR':>6} {'XGB':>6} {'Ensemble':>10} {'Best':>6}")
print("-" * 70)

while start + TRAIN_WINDOW + TEST_WINDOW <= len(X):
    train_end = start + TRAIN_WINDOW
    test_end  = train_end + TEST_WINDOW

    X_train = X[start:train_end]
    y_train = y[start:train_end]
    X_test  = X[train_end:test_end]
    y_test  = y[train_end:test_end]

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Baseline
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_s, y_train)
    base_acc = accuracy_score(y_test, dummy.predict(X_test_s))

    # Logistic regression
    lr = LogisticRegression(C=0.1, class_weight='balanced',
                             max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]
    lr_auc   = roc_auc_score(y_test, lr_proba) if len(np.unique(y_test)) > 1 else 0.5

    # XGBoost
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
        gamma=0.3, reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=scale, eval_metric='logloss', random_state=RANDOM_STATE
    )
    xgb.fit(X_train_s, y_train)
    xgb_proba = xgb.predict_proba(X_test_s)[:, 1]
    xgb_auc   = roc_auc_score(y_test, xgb_proba) if len(np.unique(y_test)) > 1 else 0.5

    # Ensemble: average probabilities 50/50
    ensemble_proba = (lr_proba + xgb_proba) / 2
    ensemble_pred  = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc   = accuracy_score(y_test, ensemble_pred)
    ensemble_auc   = roc_auc_score(y_test, ensemble_proba) if len(np.unique(y_test)) > 1 else 0.5

    best_auc  = max(lr_auc, xgb_auc, ensemble_auc)
    best_name = ['LR', 'XGB', 'Ens'][np.argmax([lr_auc, xgb_auc, ensemble_auc])]

    period = f"{dates[train_end].date()} → {dates[test_end-1].date()}"
    flag   = "GOOD" if ensemble_acc > base_acc else "BAD"

    print(f"  {fold:<5} {period:<30} "
          f"{base_acc:.3f}  {lr_auc:.3f}  {xgb_auc:.3f}  "
          f"{ensemble_auc:.3f} {flag}  {best_name}")

    lr_acc = accuracy_score(y_test, (lr_proba >= 0.5).astype(int))
    xgb_acc = accuracy_score(y_test, (xgb_proba >= 0.5).astype(int))

    results.append({
        'fold':          fold,
        'test_start':    dates[train_end].date(),
        'baseline':      base_acc,
        'lr_auc':        lr_auc,
        'xgb_auc':       xgb_auc,
        'lr_acc':        lr_acc,
        'xgb_acc':       xgb_acc,
        'lr_beats_base': lr_acc > base_acc,
        'xgb_beats_base': xgb_acc > base_acc,
        'ensemble_auc':  ensemble_auc,
        'ensemble_acc':  ensemble_acc,
        'beats_base':    ensemble_acc > base_acc
    })

    start += STEP
    fold  += 1

# Summary
r = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ENSEMBLE SUMMARY")
print("=" * 60)
print(f"\n   {'Metric':<30} {'LR':>8} {'XGBoost':>8} {'Ensemble':>10}")
print("   " + "-" * 58)
print(f"   {'Avg AUC':<30} {r['lr_auc'].mean():>8.4f} {r['xgb_auc'].mean():>8.4f} {r['ensemble_auc'].mean():>10.4f}")
print(f"   {'Avg Accuracy':<30} {'—':>8} {'—':>8} {r['ensemble_acc'].mean():>10.4f}")
n_folds = len(r)
print(
    f"   {'Folds beat baseline':<30} "
    f"{r['lr_beats_base'].sum():>3}/{n_folds}   "
    f"{r['xgb_beats_base'].sum():>3}/{n_folds}   "
    f"{r['beats_base'].sum():>3}/{n_folds}"
)
print(f"   {'Avg Baseline':<30} {r['baseline'].mean():>8.4f}")

r.to_csv('ensemble_results.csv', index=False)