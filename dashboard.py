"""
Volatility Regime Predictor — professional analytics dashboard.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from features import (
    FEATURE_COLS,
    download_market_data,
    drop_incomplete_rows,
    engineer_features,
    latest_feature_row,
)

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Vol Regime Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1400px; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 600; }
    [data-testid="stMetricLabel"] {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
        color: #8B9CB3;
    }
    .signal-high {
        background: linear-gradient(135deg, #1a2f1a 0%, #152015 100%);
        border: 1px solid #2d5a2d; border-radius: 10px; padding: 1rem 1.25rem;
    }
    .signal-low {
        background: linear-gradient(135deg, #1a2535 0%, #121820 100%);
        border: 1px solid #2a4060; border-radius: 10px; padding: 1rem 1.25rem;
    }
    .panel-caption { color: #6B7D93; font-size: 0.8rem; margin-top: 0.25rem; }
    div[data-testid="stSidebar"] { border-right: 1px solid #1e2836; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_csv(name: str) -> pd.DataFrame | None:
    path = ROOT / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "test_start" in df.columns:
        df["test_start"] = pd.to_datetime(df["test_start"])
    return df


@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame | None:
    path = ROOT / "features.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


@st.cache_resource
def load_models():
    return {
        "lr": joblib.load(ROOT / "logistic_regression_model.pkl"),
        "lr_cal": joblib.load(ROOT / "lr_calibrator.pkl"),
        "xgb": joblib.load(ROOT / "xgboost_model.pkl"),
        "xgb_cal": joblib.load(ROOT / "xgb_calibrator.pkl"),
        "scaler": joblib.load(ROOT / "scaler.pkl"),
    }


@st.cache_data(ttl=120, show_spinner="Fetching market data & running models…")
def run_live_prediction() -> dict:
    end = datetime.today()
    start = end - timedelta(days=400)
    spy, vix, vix9d, volume = download_market_data(start=start, end=end)
    df = engineer_features(spy, vix, vix9d, volume, include_target=False)
    df = drop_incomplete_rows(df, for_training=False)
    X_today, pred_date = latest_feature_row(df)
    row = df.loc[pred_date]

    models = load_models()
    X_scaled = models["scaler"].transform(X_today)
    lr_raw = models["lr"].predict_proba(X_scaled)[0][1]
    xgb_raw = models["xgb"].predict_proba(X_scaled)[0][1]
    lr_p = float(models["lr_cal"].predict_proba([[lr_raw]])[0][1])
    xgb_p = float(models["xgb_cal"].predict_proba([[xgb_raw]])[0][1])
    ens_p = (lr_p + xgb_p) / 2

    dist = abs(ens_p - 0.5)
    if dist >= 0.20:
        strength = "Strong"
    elif dist >= 0.10:
        strength = "Moderate"
    else:
        strength = "Weak"

    return {
        "date": pred_date,
        "spy_close": float(row["spy_close"]),
        "vix_close": float(row["vix_close"]),
        "realized_vol_20d": float(row["realized_vol_20d"]),
        "lr_proba": lr_p,
        "xgb_proba": xgb_p,
        "ens_proba": ens_p,
        "regime": "HIGH" if ens_p >= 0.5 else "LOW",
        "strength": strength,
        "features": X_today.iloc[0].to_dict(),
    }


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def probability_gauge(proba: float, title: str) -> go.Figure:
    regime = "High Vol" if proba >= 0.5 else "Low Vol"
    color = "#EF4444" if proba >= 0.5 else "#3B82F6"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "#1a2230",
                "steps": [
                    {"range": [0, 50], "color": "#1e3a5f"},
                    {"range": [50, 100], "color": "#3f1f1f"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 3},
                    "thickness": 0.85,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=24, r=24, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EDF4"},
    )
    fig.add_annotation(
        text=regime,
        x=0.5,
        y=0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 13, "color": color},
    )
    return fig


def model_comparison_bar(lr: float, xgb: float, ens: float) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Logistic Regression", "XGBoost", "Ensemble"],
            y=[lr, xgb, ens],
            marker_color=["#60A5FA", "#F97316", "#A78BFA"],
            text=[f"{v:.1%}" for v in [lr, xgb, ens]],
            textposition="outside",
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F59E0B", annotation_text="Decision (50%)")
    fig.update_layout(
        title="P(High Volatility Regime) by Model",
        yaxis_title="Calibrated probability",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=320,
        margin=dict(t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151B24",
        font={"color": "#E8EDF4"},
        showlegend=False,
    )
    return fig


def walk_forward_chart(wf: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    x = wf["test_start"].dt.strftime("%Y-%m-%d")
    fig.add_trace(
        go.Scatter(x=x, y=wf["lr_auc"], name="LR AUC", mode="lines+markers", line=dict(color="#60A5FA")),
    )
    fig.add_trace(
        go.Scatter(x=x, y=wf["xgb_auc"], name="XGB AUC", mode="lines+markers", line=dict(color="#F97316")),
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=wf["baseline"],
            name="Baseline acc",
            mode="lines+markers",
            line=dict(color="#EF4444", dash="dash"),
        ),
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#6B7280", annotation_text="Random (0.50)")
    fig.update_layout(
        title="Walk-Forward Out-of-Sample Performance by Fold",
        xaxis_title="Test period start",
        yaxis_title="Score",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=56, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151B24",
        font={"color": "#E8EDF4"},
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def ensemble_fold_chart(ens: pd.DataFrame) -> go.Figure:
    x = ens["test_start"].dt.strftime("%Y-%m-%d")
    fig = go.Figure()
    for col, name, color in [
        ("lr_auc", "LR", "#60A5FA"),
        ("xgb_auc", "XGB", "#F97316"),
        ("ensemble_auc", "Ensemble", "#A78BFA"),
    ]:
        fig.add_trace(go.Bar(x=x, y=ens[col], name=name, marker_color=color))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#6B7280")
    fig.update_layout(
        barmode="group",
        title="AUC by Model — Walk-Forward Folds",
        xaxis_title="Test period start",
        yaxis_title="ROC-AUC",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=56, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151B24",
        font={"color": "#E8EDF4"},
    )
    fig.update_xaxes(tickangle=-35)
    return fig


def feature_importance_chart(fi: pd.DataFrame) -> go.Figure:
    fi = fi.sort_values("importance", ascending=True)
    fig = px.bar(
        fi,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#1e3a5f", "#3B82F6"],
    )
    fig.update_layout(
        title="XGBoost Feature Importance",
        xaxis_title="Importance (gain)",
        yaxis_title="",
        height=480,
        coloraxis_showscale=False,
        margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151B24",
        font={"color": "#E8EDF4"},
    )
    return fig


def prediction_history_chart(log: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=log["date"],
            y=log["ens_proba"],
            mode="lines+markers",
            name="Ensemble P(High Vol)",
            line=dict(color="#A78BFA", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=log["date"],
            y=log["lr_proba"],
            mode="lines",
            name="LR",
            line=dict(color="#60A5FA", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=log["date"],
            y=log["xgb_proba"],
            mode="lines",
            name="XGB",
            line=dict(color="#F97316", dash="dot"),
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#F59E0B")
    fig.update_layout(
        title="Prediction Log — Calibrated Probabilities",
        xaxis_title="Date",
        yaxis_title="P(High Vol)",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151B24",
        font={"color": "#E8EDF4"},
    )
    return fig


def market_context_chart(features: pd.DataFrame) -> go.Figure | None:
    if features is None or len(features) < 30:
        return None
    recent = features.tail(126)[["vix_close", "realized_vol_20d"]].copy()
    recent = recent.rename(columns={"vix_close": "VIX", "realized_vol_20d": "Realized Vol (20d)"})
    recent.index = pd.to_datetime(recent.index)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["VIX"], name="VIX", line=dict(color="#F59E0B")))
    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["Realized Vol (20d)"] * 100,
            name="Realized Vol %",
            line=dict(color="#60A5FA"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Market Context — Last ~6 Months",
        xaxis_title="Date",
        yaxis_title="VIX",
        yaxis2=dict(title="Realized vol (ann. %)", overlaying="y", side="right"),
        height=320,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151B24",
        font={"color": "#E8EDF4"},
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Vol Regime Predictor")
    st.caption("S&P 500 · 10-day forward volatility regime")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Live Signal", "Model Performance", "Features", "History"],
        label_visibility="collapsed",
    )

    st.divider()
    refresh = st.button("Refresh live prediction", use_container_width=True, type="primary")
    if refresh:
        run_live_prediction.clear()
        st.cache_data.clear()

    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        """
        - 18 lagged features  
        - LR + XGBoost ensemble  
        - Platt-calibrated probabilities  
        - Walk-forward validated  
        """
    )
    st.caption(f"Last load: {datetime.now().strftime('%H:%M:%S')}")

# ---------------------------------------------------------------------------
# Load shared data
# ---------------------------------------------------------------------------

wf = load_csv("walk_forward_results.csv")
ens_results = load_csv("ensemble_results.csv")
pred_log = load_csv("prediction_log.csv")
fi = load_csv("xgb_feature_importance.csv")
features_df = load_features()

try:
    live = run_live_prediction()
    live_ok = True
except Exception as exc:
    live = None
    live_ok = False
    live_err = str(exc)

# Summary metrics from walk-forward
if wf is not None:
    avg_lr_auc = wf["lr_auc"].mean()
    avg_xgb_auc = wf["xgb_auc"].mean()
    avg_base = wf["baseline"].mean()
    lr_beats = int(wf["lr_beats_base"].sum())
    xgb_beats = int(wf["xgb_beats_base"].sum())
    n_folds = len(wf)
else:
    avg_lr_auc = avg_xgb_auc = avg_base = 0.0
    lr_beats = xgb_beats = n_folds = 0

if ens_results is not None:
    avg_ens_auc = ens_results["ensemble_auc"].mean()
    ens_beats = int(ens_results["beats_base"].sum())
else:
    avg_ens_auc = 0.0
    ens_beats = 0


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Volatility Regime Predictor")
    st.markdown(
        "Predict whether the S&P 500 enters a **high or low volatility regime** "
        "over the next **10 trading days**, with calibrated ensemble probabilities."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    if live_ok:
        c1.metric("Today's Regime", live["regime"] + " VOL", live["strength"])
        c2.metric("Ensemble P(High Vol)", f"{live['ens_proba']:.1%}")
        c3.metric("SPY", f"${live['spy_close']:,.2f}")
        c4.metric("VIX", f"{live['vix_close']:.1f}")
        c5.metric("Realized Vol (20d)", f"{live['realized_vol_20d']:.1%}")
    else:
        c1.warning("Live prediction unavailable")

    st.divider()

    left, right = st.columns([1.1, 1])
    with left:
        if live_ok:
            st.plotly_chart(
                probability_gauge(live["ens_proba"], "Ensemble — High Vol Probability"),
                use_container_width=True,
            )
            st.plotly_chart(model_comparison_bar(live["lr_proba"], live["xgb_proba"], live["ens_proba"]),
                            use_container_width=True)
        else:
            st.error(f"Could not load models: {live_err}")

    with right:
        if live_ok:
            css = "signal-high" if live["regime"] == "HIGH" else "signal-low"
            if live["regime"] == "HIGH":
                strategy = (
                    "**Long volatility** — straddles, strangles, debit spreads. "
                    "Elevated IV expansion expected."
                )
            else:
                strategy = (
                    "**Short volatility** — iron condors, covered calls, credit spreads. "
                    "Theta decay in calm regimes."
                )
            st.markdown(
                f'<div class="{css}"><h4 style="margin:0">Options Signal</h4>'
                f'<p style="margin:0.5rem 0 0">{strategy}</p></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<p class="panel-caption">Based on ensemble probability vs 50% threshold</p>',
                        unsafe_allow_html=True)

        st.markdown("#### Walk-Forward Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Ensemble AUC", f"{avg_ens_auc:.3f}" if ens_results is not None else "—")
        m2.metric("LR beats baseline", f"{lr_beats}/{n_folds}" if wf is not None else "—")
        m3.metric("Ensemble beats baseline", f"{ens_beats}/{n_folds}" if ens_results is not None else "—")

        if wf is not None:
            st.dataframe(
                wf[["fold", "test_start", "test_end", "lr_auc", "xgb_auc", "lr_beats_base"]].rename(
                    columns={
                        "test_start": "Period start",
                        "test_end": "Period end",
                        "lr_auc": "LR AUC",
                        "xgb_auc": "XGB AUC",
                        "lr_beats_base": "LR > base",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    if features_df is not None:
        ctx = market_context_chart(features_df)
        if ctx:
            st.plotly_chart(ctx, use_container_width=True)

elif page == "Live Signal":
    st.title("Live Signal")
    if not live_ok:
        st.error(live_err)
        st.stop()

    st.caption(f"Prediction date: **{live['date'].date()}** · features through prior close (no lookahead)")

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(probability_gauge(live["lr_proba"], "Logistic Regression"), use_container_width=True)
    g2.plotly_chart(probability_gauge(live["xgb_proba"], "XGBoost"), use_container_width=True)
    g3.plotly_chart(probability_gauge(live["ens_proba"], "Ensemble"), use_container_width=True)

    st.plotly_chart(
        model_comparison_bar(live["lr_proba"], live["xgb_proba"], live["ens_proba"]),
        use_container_width=True,
    )

    st.subheader("Today's feature vector")
    feat_tbl = pd.DataFrame(
        {"Feature": FEATURE_COLS, "Value": [live["features"].get(c, np.nan) for c in FEATURE_COLS]}
    )
    st.dataframe(
        feat_tbl.style.format({"Value": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Out-of-sample **walk-forward** validation — models trained only on past data.")

    if wf is None:
        st.warning("Run `python walk_forward.py` to generate walk_forward_results.csv")
    else:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Avg LR AUC", f"{avg_lr_auc:.3f}")
        a2.metric("Avg XGB AUC", f"{avg_xgb_auc:.3f}")
        a3.metric("Avg Ensemble AUC", f"{avg_ens_auc:.3f}")
        a4.metric("Avg Baseline Acc", f"{avg_base:.1%}")

        st.plotly_chart(walk_forward_chart(wf), use_container_width=True)

    if ens_results is not None:
        st.plotly_chart(ensemble_fold_chart(ens_results), use_container_width=True)

        st.subheader("Fold-level results")
        display = ens_results.copy()
        display["test_start"] = display["test_start"].dt.date
        display["beats_base"] = display["beats_base"].map({True: "Yes", False: "No"})
        st.dataframe(
            display.round(4),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Run `python ensemble.py` for ensemble fold breakdown.")

elif page == "Features":
    st.title("Feature Analysis")
    if fi is not None:
        st.plotly_chart(feature_importance_chart(fi), use_container_width=True)
        st.dataframe(fi.sort_values("importance", ascending=False), hide_index=True, use_container_width=True)
    else:
        st.warning("Run `python train_xgboost.py` to generate xgb_feature_importance.csv")

    if features_df is not None and "target" in features_df.columns:
        st.subheader("Feature ↔ target correlation (full history)")
        corrs = features_df[FEATURE_COLS].corrwith(features_df["target"]).sort_values(ascending=False)
        corr_df = corrs.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        fig = px.bar(
            corr_df,
            x="Correlation",
            y="Feature",
            orientation="h",
            color="Correlation",
            color_continuous_scale=["#3B82F6", "#EF4444"],
        )
        fig.update_layout(
            height=500,
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#151B24",
            font={"color": "#E8EDF4"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python create_features.py` for correlation analysis.")

elif page == "History":
    st.title("Prediction History")
    if pred_log is None or pred_log.empty:
        st.info("No predictions logged yet. Run `python predict_today.py` or refresh live signal.")
    else:
        pred_log = pred_log.sort_values("date")
        st.plotly_chart(prediction_history_chart(pred_log), use_container_width=True)

        pred_log_display = pred_log.copy()
        pred_log_display["date"] = pred_log_display["date"].dt.date
        pred_log_display["ens_proba"] = pred_log_display["ens_proba"].map(lambda x: f"{x:.1%}")
        pred_log_display["lr_proba"] = pred_log_display["lr_proba"].map(lambda x: f"{x:.1%}")
        pred_log_display["xgb_proba"] = pred_log_display["xgb_proba"].map(lambda x: f"{x:.1%}")
        st.dataframe(pred_log_display, use_container_width=True, hide_index=True)
