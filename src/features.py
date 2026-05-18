"""Feature engineering shared by training and inference pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ["temp_24h", "precip_24h", "windspeed_24h"]

LAG_HOURS = [1, 3, 6, 12, 24]
ROLL_WINDOWS = [6, 24, 168]


def build_features(df: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
    """Build feature matrix with lag, rolling, cyclical, and target columns.

    Args:
        df: DataFrame with columns matching the weather SQLite schema:
            time, temp, humidity, pressure, windspeed, precipitation, weathercode
        inference: When True, skip target columns and only drop NaN in features.
            Use this for predict.py; leave False for train.py.

    Returns:
        Feature-engineered DataFrame. In training mode includes target columns
        temp_24h, precip_24h, windspeed_24h with NaN rows dropped.
        In inference mode returns all rows with valid feature values only.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # --- lag features ---
    for col, prefix in [("temp", "temp"), ("windspeed", "windspeed"), ("precipitation", "precip")]:
        for h in LAG_HOURS:
            df[f"{prefix}_lag_{h}h"] = df[col].shift(h)

    # --- rolling stats on temperature ---
    for w in ROLL_WINDOWS:
        df[f"temp_roll_mean_{w}h"] = df["temp"].shift(1).rolling(w).mean()
        df[f"temp_roll_std_{w}h"] = df["temp"].shift(1).rolling(w).std()

    # --- cyclical time encoding ---
    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["time"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["time"].dt.month / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofweek / 7)

    # --- boolean flags ---
    df["is_weekend"] = (df["time"].dt.dayofweek >= 5).astype(int)
    df["is_daytime"] = ((df["time"].dt.hour >= 6) & (df["time"].dt.hour <= 22)).astype(int)

    # --- delta features ---
    for h in [1, 3, 6]:
        df[f"temp_delta_{h}h"] = df["temp"] - df[f"temp_lag_{h}h"]

    feature_cols = _feature_columns(df)

    if inference:
        # drop only rows where features are incomplete (lag warmup period)
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
    else:
        # --- targets (shift -24 = value 24 hours ahead) ---
        df["temp_24h"] = df["temp"].shift(-24)
        df["precip_24h"] = df["precipitation"].shift(-24)
        df["windspeed_24h"] = df["windspeed"].shift(-24)
        # drop rows with NaN in any feature or target
        df = df.dropna(subset=feature_cols + TARGETS).reset_index(drop=True)

    return df


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of all feature column names (excluding targets and raw source cols)."""
    exclude = {"time", "temp", "humidity", "pressure", "windspeed", "precipitation",
               "weathercode"} | set(TARGETS)
    return [c for c in df.columns if c not in exclude]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Public accessor for the feature column list after build_features() is applied."""
    return _feature_columns(df)
