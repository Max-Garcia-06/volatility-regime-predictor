"""Threshold analysis for the saved logistic regression model."""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from features import FEATURE_COLS, chronological_split

warnings.filterwarnings("ignore")

df = pd.read_csv("features.csv", index_col=0, parse_dates=True)
X = df[FEATURE_COLS]
y = df["target"]

_, _, _, _, X_test, y_test = chronological_split(X, y)

model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

X_test_scaled = scaler.transform(X_test)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

print("=" * 60)
print("THRESHOLD ANALYSIS (Logistic Regression)")
print("=" * 60)
print(f"\n{'Threshold':>10} {'Accuracy':>10} {'F1':>10} {'Pos%':>10}")
print("-" * 45)

best_f1 = 0.0
best_threshold = 0.5

for threshold in np.arange(0.3, 0.8, 0.05):
    preds = (y_test_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)
    pos_pct = preds.mean() * 100
    flag = " ← current" if abs(threshold - 0.5) < 0.01 else ""
    print(f"   {threshold:.2f}    {acc:.4f}    {f1:.4f}    {pos_pct:.1f}%{flag}")
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest Threshold: {best_threshold:.2f}  (F1 = {best_f1:.4f})")
print(f"\nTest AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
