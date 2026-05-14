"""
Feature Engineering
====================
Builds lag features, rolling statistics, and calendar features
for the LightGBM quantile regression model.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_time_features(df: pd.DataFrame, date_col: str = "Datetime") -> pd.DataFrame:
    """Calendar-based features capturing seasonality patterns."""
    dt = pd.to_datetime(df[date_col])
    
    df = df.copy()
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_business_hour"] = ((dt.dt.hour >= 8) & (dt.dt.hour <= 18)).astype(int)
    
    # Cyclical encodings — prevent discontinuity at period boundaries
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    lags: List[int] = [1, 2, 3, 6, 12, 24, 48, 168],
) -> pd.DataFrame:
    """
    Autoregressive lag features.
    lag=168 captures same-hour-last-week — critical for weekly seasonality.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    windows: List[int] = [6, 24, 168],
) -> pd.DataFrame:
    """Rolling mean, std, min, max at multiple time scales."""
    df = df.copy()
    for w in windows:
        col_prefix = f"roll_{w}h"
        df[f"{col_prefix}_mean"] = df[target_col].shift(1).rolling(w, min_periods=w // 2).mean()
        df[f"{col_prefix}_std"] = df[target_col].shift(1).rolling(w, min_periods=w // 2).std()
        df[f"{col_prefix}_min"] = df[target_col].shift(1).rolling(w, min_periods=w // 2).min()
        df[f"{col_prefix}_max"] = df[target_col].shift(1).rolling(w, min_periods=w // 2).max()
    return df


def add_diff_features(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    periods: List[int] = [1, 24, 168],
) -> pd.DataFrame:
    """First differences — help with non-stationarity."""
    df = df.copy()
    for p in periods:
        df[f"diff_{p}h"] = df[target_col].diff(p)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "PJME_MW",
    date_col: str = "Datetime",
    lag_hours: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns DataFrame with features and target, NaN rows dropped.
    """
    lag_hours = lag_hours or [1, 2, 3, 6, 12, 24, 48, 168]
    rolling_windows = rolling_windows or [6, 24, 168]
    
    df = add_time_features(df, date_col)
    df = add_lag_features(df, target_col, lag_hours)
    df = add_rolling_features(df, target_col, rolling_windows)
    df = add_diff_features(df, target_col)
    
    # Drop rows with NaN from lag creation (first max_lag rows)
    max_lag = max(lag_hours) if lag_hours else 168
    df = df.iloc[max_lag:].reset_index(drop=True)
    df = df.dropna()
    
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "PJME_MW", date_col: str = "Datetime") -> List[str]:
    """Return feature column names (excludes target and datetime)."""
    exclude = {target_col, date_col, "Datetime"}
    return [c for c in df.columns if c not in exclude]


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple:
    """Temporal split — no shuffling, preserves time ordering."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    
    print(f"Train: {len(train):,} rows ({train['Datetime'].min().date()} → {train['Datetime'].max().date()})")
    print(f"Val:   {len(val):,} rows ({val['Datetime'].min().date()} → {val['Datetime'].max().date()})")
    print(f"Test:  {len(test):,} rows ({test['Datetime'].min().date()} → {test['Datetime'].max().date()})")
    
    return train, val, test
