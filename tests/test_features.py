"""Tests for the feature engineering module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import TARGETS, build_features, get_feature_columns

EXPECTED_FEATURE_GROUPS = [
    "temp_lag_1h", "temp_lag_3h", "temp_lag_6h", "temp_lag_12h", "temp_lag_24h",
    "windspeed_lag_1h", "windspeed_lag_24h",
    "precip_lag_1h", "precip_lag_24h",
    "temp_roll_mean_6h", "temp_roll_std_6h",
    "temp_roll_mean_24h", "temp_roll_std_168h",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dayofweek_sin", "dayofweek_cos",
    "is_weekend", "is_daytime",
    "temp_delta_1h", "temp_delta_3h", "temp_delta_6h",
    "cloudcover_raw", "radiation_raw", "vpd_raw", "dewpoint_spread",
    "cloudcover_lag_1h", "cloudcover_lag_24h", "cloudcover_roll_mean_6h",
    "vpd_lag_1h", "vpd_lag_6h",
]


def _make_df(n: int = 300) -> pd.DataFrame:
    """Create a synthetic weather DataFrame with n hourly rows."""
    times = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "time": times.astype(str),
        "temp": rng.uniform(-5, 35, n),
        "humidity": rng.uniform(30, 100, n),
        "pressure": rng.uniform(990, 1030, n),
        "windspeed": rng.uniform(0, 20, n),
        "precipitation": rng.uniform(0, 10, n),
        "weathercode": rng.integers(0, 100, n),
        "cloudcover": rng.uniform(0, 100, n),
        "dew_point": rng.uniform(-10, 30, n),
        "shortwave_radiation": rng.uniform(0, 900, n),
        "vapour_pressure_deficit": rng.uniform(0, 3, n),
    })


def test_expected_feature_columns_present() -> None:
    df = build_features(_make_df())
    feature_cols = get_feature_columns(df)
    for col in EXPECTED_FEATURE_GROUPS:
        assert col in feature_cols, f"Missing feature column: {col}"


def test_target_columns_present() -> None:
    df = build_features(_make_df())
    for target in TARGETS:
        assert target in df.columns, f"Missing target column: {target}"


def test_no_nan_in_features_or_targets() -> None:
    df = build_features(_make_df())
    feature_cols = get_feature_columns(df)
    for col in feature_cols + TARGETS:
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"NaN found in column: {col} ({nan_count} rows)"


def test_target_shift_is_correct() -> None:
    """temp_24h at row i should equal temp at row i+24 of the original data."""
    raw = _make_df(200)
    df = build_features(raw)

    # pick a row that is safely within range
    row_idx = 0
    ts = pd.to_datetime(df["time"].iloc[row_idx])
    expected_ts = ts + pd.Timedelta(hours=24)

    raw_sorted = raw.copy()
    raw_sorted["time"] = pd.to_datetime(raw_sorted["time"])
    raw_sorted = raw_sorted.sort_values("time")

    future_rows = raw_sorted[raw_sorted["time"] == expected_ts]
    if future_rows.empty:
        pytest.skip("Target timestamp not in synthetic data range")

    expected_temp = future_rows["temp"].iloc[0]
    actual_target = df["temp_24h"].iloc[row_idx]
    assert abs(actual_target - expected_temp) < 1e-6, (
        f"temp_24h mismatch: expected {expected_temp}, got {actual_target}"
    )


def test_output_row_count_is_less_than_input() -> None:
    """build_features should drop rows with NaN (lag warmup + target tail)."""
    raw = _make_df(300)
    df = build_features(raw)
    assert len(df) < len(raw), "Expected fewer rows after dropping NaN"
    assert len(df) > 0, "Output DataFrame should not be empty"
