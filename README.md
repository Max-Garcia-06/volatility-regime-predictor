# Volatility Regime Predictor

End-to-end ML pipeline that predicts whether the S&P 500 will enter a
**high or low volatility regime** over the next 10 trading days — and runs
a fresh prediction every day with a live options strategy signal.

Built with Logistic Regression + XGBoost ensemble, validated with
walk-forward cross-validation across 10 out-of-sample periods spanning 2023–2026.

**Avg Ensemble AUC: 0.730 | Beats baseline in 9/10 walk-forward folds | +20pp accuracy improvement over baseline**

---

## Results (Walk-Forward Validation)

> Walk-forward validation simulates real trading: models are trained only on past data
> and evaluated on future unseen periods. No lookahead bias.

| Model | Avg AUC | Avg Accuracy | Folds Beat Baseline |
|-------|---------|--------------|---------------------|
| Logistic Regression | 0.735 | 65.5% | 8 / 10 |
| XGBoost | 0.649 | 61.8% | 7 / 10 |
| **Ensemble** | **0.730** | **62.7%** | **9 / 10** |

Avg baseline (predict majority class): **45.0%**

---

## Key Findings

- **Direction prediction doesn't work** (AUC ≈ 0.50) — consistent with EMH
- **Volatility regime prediction does** (Avg AUC 0.730) — vol clusters, and
  VIX-based features capture this reliably
- **Strongest signal during stress** — Ensemble AUC hit 0.908 during the
  Feb–May 2025 tariff-driven volatility spike
- **Known edge case** — models lag after sudden vol collapse (post-stress
  calm periods), a structural challenge for any regime model

---

## Daily Prediction

Running `predict_today.py` downloads fresh market data, computes all features,
and outputs a calibrated probability with an options strategy signal:

```
TODAY'S VOLATILITY REGIME PREDICTION

Run time: 2026-04-13 13:26:26

Prediction for: 2026-04-13
   SPY Close:   $686.13
   VIX Level:   19.18
   Realized Vol (20d): 19.7%

Model Probabilities (High Vol):
   Logistic Regression: 0.327
   XGBoost:             0.163
   Ensemble:            0.245

Predicted Regime:   LOW VOLATILITY
   High-vol probability: 24.5%  [Strong]

Options Implication:
   → Consider SHORT options (iron condors, covered calls)
   → Low vol expected — collect theta decay
```

Predictions are logged to `prediction_log.csv` with deduplication on same-day re-runs.

---

## Trading Application

| Signal | Strategy | Rationale |
|--------|----------|-----------|
| High vol regime predicted | Long options (straddles, strangles) | Benefit from IV expansion |
| Low vol regime predicted | Short options (iron condors, covered calls) | Collect theta in calm markets |

Signal strength is reported as **Strong / Moderate / Weak** based on distance
from the 0.5 decision boundary, so position sizing can be adjusted accordingly.

---

## Features (18 total)

| Category | Features |
|----------|----------|
| **Price Momentum** | 1d, 5d, 10d, 20d SPY returns |
| **Technical** | RSI-14, Bollinger Band z-score |
| **Trend** | Price vs MA50, Price vs MA200 |
| **VIX** | Level, 1d return, MA10/MA20 ratio, 60d percentile, spike flag |
| **VIX Term Structure** | VIX9D / VIX ratio — backwardation signals near-term fear |
| **Vol Risk Premium** | VIX − realized vol — measures excess fear pricing |
| **Realized Volatility** | 10d and 20d annualized realized vol |
| **Volume** | Volume ratio vs 20d average |

Top features by XGBoost importance: `vix_percentile`, `vol_risk_premium`,
`price_vs_ma50`, `spy_return_10d`, `realized_vol_10d`

---

## Probability Calibration

Raw model probabilities are post-processed with **Platt scaling** (logistic
regression fitted on the validation set). This maps model output to empirically
grounded probabilities — P=0.25 means roughly 25% of historically similar
days resulted in a high-vol regime, making the output directly useful for
position sizing.

---

## Project Structure

```
volatility-regime-predictor/
├── predict_today.py         # Live daily prediction with options signal
├── create_features.py       # Feature engineering + volatility target (18 features)
├── train_model.py           # Logistic Regression + Platt calibration
├── train_xgboost.py         # XGBoost + Platt calibration
├── walk_forward.py          # Walk-forward validation (10 folds)
├── ensemble.py              # Ensemble evaluation
├── collect_data.py          # Downloads SPY options chain data
├── diagnostic.py            # Threshold analysis on saved model
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt

# Run a prediction immediately (models are pre-trained and included)
python predict_today.py

# To retrain on fresh data
python create_features.py   # regenerates features.csv
python train_model.py       # retrains LR + calibrator
python train_xgboost.py     # retrains XGBoost + calibrator

# To re-run validation
python walk_forward.py
python ensemble.py
```

---

## Tech Stack

`Python` `XGBoost` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `yFinance`
