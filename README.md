# Volatility Regime Predictor

End-to-end ML pipeline for predicting S&P 500 volatility regimes using
Logistic Regression, XGBoost, and an Ensemble model — validated with
walk-forward cross-validation across 10 out-of-sample periods.

**Avg Ensemble AUC: 0.722 | Beats baseline in 9/10 walk-forward folds**

---

## Results

| Model | Avg AUC | Avg Accuracy | Folds Beat Baseline |
|-------|---------|--------------|---------------------|
| Logistic Regression | 0.688 | 69.3% | 9 / 10 |
| XGBoost | 0.713 | 65.2% | 7 / 10 |
| **Ensemble** | **0.722** | **67.8%** | **8 / 10** |

---

## Key Findings

- **Direction prediction failed** (AUC ≈ 0.49) — consistent with the
  Efficient Market Hypothesis
- **Volatility regime prediction works** (Avg AUC 0.722) — volatility
  clusters, and VIX-based features capture this signal
- **Best signal during market stress** — Ensemble AUC hit 0.974 during
  the April 2025 tariff-driven volatility spike
- **Known limitation** — models struggle after sudden volatility collapse
  (Summer 2025 post-tariff calm)

---

## Trading Application

| Signal | Strategy | Rationale |
|--------|----------|-----------|
| High vol regime predicted | Long options (straddles, strangles) | Benefit from IV expansion |
| Low vol regime predicted | Short options (iron condors, covered calls) | Collect theta in calm markets |

---

## Project Structure

```
volatility-regime-predictor/
├── collect_data.py          # Downloads SPY + VIX data via yFinance
├── create_features.py       # Feature engineering + volatility target
├── train_model.py           # Logistic Regression training + evaluation
├── train_xgboost.py         # XGBoost training + evaluation
├── walk_forward.py          # Walk-forward validation (LR vs XGBoost)
├── ensemble.py              # Ensemble model (averaged probabilities)
├── requirements.txt
└── README.md
```

---

## Features Used

| Category | Features |
|----------|----------|
| **Price Momentum** | 1d, 5d, 10d, 20d returns |
| **Technical** | RSI-14, Bollinger Band position |
| **Trend** | Price vs MA50, Price vs MA200 |
| **Volatility** | Realized vol 10d & 20d |
| **VIX** | VIX level, 1d return, MA ratio, 60d percentile |
| **Volume** | Volume ratio vs 20d average |

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline in order
python collect_data.py
python create_features.py
python train_model.py
python train_xgboost.py
python walk_forward.py
python ensemble.py
```

---

## Tech Stack

`Python` `XGBoost` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `yFinance`
