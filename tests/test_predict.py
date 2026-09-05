"""Tests for the predict module using mocked SQLite data."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _make_weather_rows(n: int, start: str = "2023-06-01") -> list[tuple]:
    times = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(7)
    return [
        (
            str(t)[:16],
            float(rng.uniform(15, 30)),
            float(rng.uniform(40, 90)),
            float(rng.uniform(1000, 1020)),
            float(rng.uniform(1, 15)),
            float(rng.uniform(0, 5)),
            int(rng.integers(0, 3)),
            float(rng.uniform(0, 100)),
            float(rng.uniform(10, 28)),
            float(rng.uniform(0, 900)),
            float(rng.uniform(0, 3)),
        )
        for t in times
    ]


def _make_forecast_rows(n: int, start: str) -> list[tuple]:
    times = pd.date_range(start, periods=n, freq="h")
    rng = np.random.default_rng(13)
    return [
        (
            str(t)[:16],
            float(rng.uniform(15, 30)),
            float(rng.uniform(40, 90)),
            float(rng.uniform(1000, 1020)),
            float(rng.uniform(1, 15)),
            float(rng.uniform(0, 5)),
            int(rng.integers(0, 3)),
            float(rng.uniform(0, 100)),
            float(rng.uniform(10, 28)),
            float(rng.uniform(0, 900)),
            float(rng.uniform(0, 3)),
        )
        for t in times
    ]


CREATE_WEATHER = """
    CREATE TABLE weather (
        time TEXT PRIMARY KEY, temp REAL, humidity REAL,
        pressure REAL, windspeed REAL, precipitation REAL, weathercode INTEGER,
        cloudcover REAL, dew_point REAL, shortwave_radiation REAL,
        vapour_pressure_deficit REAL
    )
"""
CREATE_FORECAST = """
    CREATE TABLE forecast_raw (
        time TEXT PRIMARY KEY, temp REAL, humidity REAL,
        pressure REAL, windspeed REAL, precipitation REAL, weathercode INTEGER,
        cloudcover REAL, dew_point REAL, shortwave_radiation REAL,
        vapour_pressure_deficit REAL
    )
"""
INSERT_SQL = "INSERT INTO {t} VALUES (?,?,?,?,?,?,?,?,?,?,?)"


@pytest.fixture()
def mock_db(tmp_path: Path) -> Path:
    """SQLite DB with 300 weather rows + 48 forecast_raw rows."""
    db = tmp_path / "weather.db"
    conn = sqlite3.connect(db)
    conn.execute(CREATE_WEATHER)
    conn.execute(CREATE_FORECAST)

    weather_rows = _make_weather_rows(300, "2023-01-01")
    conn.executemany(INSERT_SQL.format(t="weather"), weather_rows)

    # forecast starts right after last weather row
    last_weather_ts = weather_rows[-1][0]
    forecast_start = str(pd.to_datetime(last_weather_ts) + pd.Timedelta(hours=1))[:16]
    forecast_rows = _make_forecast_rows(48, forecast_start)
    conn.executemany(INSERT_SQL.format(t="forecast_raw"), forecast_rows)

    conn.commit()
    conn.close()
    return db


@pytest.fixture()
def mock_models_dir(tmp_path: Path, mock_db: Path) -> Path:
    """Train tiny stub models and write metadata into a temp models dir."""
    from sklearn.dummy import DummyRegressor

    import joblib
    from features import build_features, get_feature_columns

    conn = sqlite3.connect(mock_db)
    raw = pd.read_sql("SELECT * FROM weather ORDER BY time", conn)
    conn.close()

    df = build_features(raw)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    target_files = {
        "temp_24h": "xgb_temp.pkl",
        "precip_24h": "xgb_precip.pkl",
        "windspeed_24h": "xgb_windspeed.pkl",
    }
    for target, fname in target_files.items():
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X, df[target])
        joblib.dump(dummy, models_dir / fname)

    metadata = {
        "trained_at": "2023-06-15T10:00:00",
        "city": "Pattaya",
        "lat": 12.92,
        "lon": 100.88,
        "train_rows": len(df),
        "test_rows": 0,
        "feature_columns": feature_cols,
        "mae_temp": 1.5,
        "mae_precip": 0.8,
        "mae_wind": 1.2,
        "rmse_temp": 2.0,
        "r2_temp": 0.91,
        "production_ready": True,
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return models_dir


def test_predict_returns_24_rows(mock_db: Path, mock_models_dir: Path, tmp_path: Path) -> None:
    output_csv = tmp_path / "latest_predictions.csv"

    with (
        patch("predict.MODELS_DIR", mock_models_dir),
        patch("predict.OUTPUT_PATH", output_csv),
        patch.dict("os.environ", {"DB_PATH": str(mock_db), "CITY": "Pattaya"}),
    ):
        import predict
        result = predict.predict()

    assert len(result) == 24, f"Expected 24 rows, got {len(result)}"


def test_predict_output_columns(mock_db: Path, mock_models_dir: Path, tmp_path: Path) -> None:
    output_csv = tmp_path / "latest_predictions.csv"

    with (
        patch("predict.MODELS_DIR", mock_models_dir),
        patch("predict.OUTPUT_PATH", output_csv),
        patch.dict("os.environ", {"DB_PATH": str(mock_db), "CITY": "Pattaya"}),
    ):
        import predict
        result = predict.predict()

    expected = {"timestamp", "pred_temp", "pred_precip", "pred_windspeed", "temp_lower", "temp_upper"}
    assert set(result.columns) == expected, f"Unexpected columns: {set(result.columns)}"


def test_predict_saves_csv(mock_db: Path, mock_models_dir: Path, tmp_path: Path) -> None:
    output_csv = tmp_path / "predictions.csv"

    with (
        patch("predict.MODELS_DIR", mock_models_dir),
        patch("predict.OUTPUT_PATH", output_csv),
        patch.dict("os.environ", {"DB_PATH": str(mock_db), "CITY": "Pattaya"}),
    ):
        import predict
        predict.predict()

    assert output_csv.exists(), "CSV file was not saved"
    saved = pd.read_csv(output_csv)
    assert len(saved) == 24


def test_predict_no_negative_precip(mock_db: Path, mock_models_dir: Path, tmp_path: Path) -> None:
    output_csv = tmp_path / "pred.csv"

    with (
        patch("predict.MODELS_DIR", mock_models_dir),
        patch("predict.OUTPUT_PATH", output_csv),
        patch.dict("os.environ", {"DB_PATH": str(mock_db), "CITY": "Pattaya"}),
    ):
        import predict
        result = predict.predict()

    assert (result["pred_precip"] >= 0).all(), "Negative precipitation values found"
    assert (result["pred_windspeed"] >= 0).all(), "Negative wind speed values found"
