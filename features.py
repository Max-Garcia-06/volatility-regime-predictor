"""Shared feature definitions and engineering for train and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

FEATURE_COLS: list[str] = [
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_10d",
    "spy_return_20d",
    "rsi_14",
    "bb_position",
    "price_vs_ma50",
    "price_vs_ma200",
    "vix_level",
    "vix_return_1d",
    "vix_ma_ratio",
    "vix_percentile",
    "vix_spike",
    "vix_term_structure",
    "vol_risk_premium",
    "realized_vol_10d",
    "realized_vol_20d",
    "volume_ratio",
]

TARGET_HORIZON = 10
TARGET_ROLLING_MEDIAN = 60

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

WALK_FORWARD_TRAIN = 400
WALK_FORWARD_TEST = 60
WALK_FORWARD_STEP = 60

RANDOM_STATE = 42


def download_market_data(
    *,
    period: str | None = "5y",
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Download SPY, VIX, VIX9D, and volume; return aligned Close/Volume series."""
    kwargs: dict = {"progress": False, "auto_adjust": True}
    if period is not None:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end

    spy_raw = yf.download("SPY", **kwargs)
    vix_raw = yf.download("^VIX", **kwargs)
    vix9d_raw = yf.download("^VIX9D", **kwargs)

    for frame in (spy_raw, vix_raw, vix9d_raw):
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

    spy = spy_raw["Close"].squeeze()
    volume = spy_raw["Volume"].squeeze()
    vix = vix_raw["Close"].squeeze()
    vix9d = vix9d_raw["Close"].squeeze()

    idx = spy.index.intersection(vix.index).intersection(vix9d.index)
    return spy.loc[idx], vix.loc[idx], vix9d.loc[idx], volume.loc[idx]


def engineer_features(
    spy: pd.Series,
    vix: pd.Series,
    vix9d: pd.Series,
    volume: pd.Series,
    *,
    include_target: bool = False,
) -> pd.DataFrame:
    """
    Build model features aligned with training.

    All predictors are shifted by one day so row t only uses information
    available through the close of t-1 (no lookahead).
    """
    df = pd.DataFrame(
        {
            "spy_close": spy,
            "volume": volume,
            "vix_close": vix,
            "vix9d_close": vix9d,
        },
        index=spy.index,
    )

    if include_target:
        future_vol = (
            df["spy_close"].pct_change().rolling(TARGET_HORIZON).std().shift(-TARGET_HORIZON)
            * np.sqrt(252)
        )
        df["future_vol_10d"] = future_vol
        df["target"] = (
            future_vol > future_vol.rolling(TARGET_ROLLING_MEDIAN).median()
        ).astype(int)

    # Price momentum
    df["spy_return_1d"] = df["spy_close"].pct_change(1).shift(1)
    df["spy_return_5d"] = df["spy_close"].pct_change(5).shift(1)
    df["spy_return_10d"] = df["spy_close"].pct_change(10).shift(1)
    df["spy_return_20d"] = df["spy_close"].pct_change(20).shift(1)

    # RSI
    delta = df["spy_close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = (100 - (100 / (1 + gain / loss))).shift(1)

    # Bollinger bands
    ma20 = df["spy_close"].rolling(20).mean()
    std20 = df["spy_close"].rolling(20).std()
    df["bb_position"] = ((df["spy_close"] - ma20) / (2 * std20)).shift(1)

    # Price vs moving averages
    df["price_vs_ma50"] = (df["spy_close"] / df["spy_close"].rolling(50).mean() - 1).shift(1)
    df["price_vs_ma200"] = (df["spy_close"] / df["spy_close"].rolling(200).mean() - 1).shift(
        1
    )

    # VIX
    df["vix_level"] = df["vix_close"].shift(1)
    df["vix_return_1d"] = df["vix_close"].pct_change(1).shift(1)
    vix_ma10 = df["vix_close"].rolling(10).mean().shift(1)
    vix_ma20 = df["vix_close"].rolling(20).mean().shift(1)
    df["vix_ma_10"] = vix_ma10
    df["vix_ma_20"] = vix_ma20
    df["vix_ma_ratio"] = vix_ma10 / vix_ma20
    df["vix_percentile"] = df["vix_close"].rolling(60).rank(pct=True).shift(1)
    df["vix_spike"] = (
        (df["vix_close"] > df["vix_close"].rolling(20).mean() * 1.2).astype(int).shift(1)
    )

    # VIX term structure and vol risk premium
    df["vix_term_structure"] = (df["vix9d_close"] / df["vix_close"]).shift(1)
    realized_vol_20d_raw = df["spy_close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["vol_risk_premium"] = (df["vix_close"] / 100 - realized_vol_20d_raw).shift(1)
    df["realized_vol_10d"] = (
        df["spy_close"].pct_change().rolling(10).std() * np.sqrt(252)
    ).shift(1)
    df["realized_vol_20d"] = realized_vol_20d_raw.shift(1)

    # Volume
    df["volume_ratio"] = (df["volume"] / df["volume"].rolling(20).mean()).shift(1)

    return df


def drop_incomplete_rows(df: pd.DataFrame, *, for_training: bool) -> pd.DataFrame:
    """Drop rows with missing values in required columns."""
    required = FEATURE_COLS.copy()
    if for_training:
        required.append("target")
    return df.dropna(subset=required)


def latest_feature_row(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return the most recent complete feature vector (1-row DataFrame) and its date."""
    complete = df[FEATURE_COLS].dropna()
    if complete.empty:
        raise ValueError("No complete feature rows available; download more history.")
    date = complete.index[-1]
    return complete.iloc[[-1]], date


def chronological_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Time-ordered train / validation / test split."""
    n = len(X)
    train_end = int(TRAIN_FRAC * n)
    val_end = train_end + int(VAL_FRAC * n)
    return (
        X.iloc[:train_end],
        y.iloc[:train_end],
        X.iloc[train_end:val_end],
        y.iloc[train_end:val_end],
        X.iloc[val_end:],
        y.iloc[val_end:],
    )
