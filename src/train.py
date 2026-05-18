"""Train one XGBoost model per target, evaluate, and save with metadata."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from features import TARGETS, build_features, get_feature_columns

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PRODUCTION_MAE_THRESHOLD = 2.0  # °C

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "early_stopping_rounds": 50,
    "eval_metric": "mae",
    "random_state": 42,
    "n_jobs": -1,
}


def _load_weather(db_path: str) -> pd.DataFrame:
    """Load full weather table from SQLite."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM weather ORDER BY time", conn)
    conn.close()
    log.info("Loaded %d rows from weather table", len(df))
    return df


def _chronological_split(df: pd.DataFrame, test_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test preserving time order. Test = last test_days days."""
    df["time"] = pd.to_datetime(df["time"])
    cutoff = df["time"].max() - pd.Timedelta(days=test_days)
    train = df[df["time"] <= cutoff].copy()
    test = df[df["time"] > cutoff].copy()
    log.info("Train rows: %d  Test rows: %d  Cutoff: %s", len(train), len(test), cutoff.date())
    return train, test


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, skipping zeros in y_true."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _train_one(
    target: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[XGBRegressor, dict]:
    """Train one XGBoost model and compute evaluation metrics."""
    log.info("Training model for target: %s", target)
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = _mape(y_test.values, y_pred)

    metrics = {
        "mae": float(round(mae, 4)),
        "rmse": float(round(rmse, 4)),
        "r2": float(round(r2, 4)),
        "mape": float(round(mape, 2)),
    }
    log.info("%s  MAE=%.3f  RMSE=%.3f  R2=%.3f  MAPE=%.1f%%", target, mae, rmse, r2, mape)
    return model, metrics


def _print_feature_importances(model: XGBRegressor, feature_cols: list[str], target: str, top_n: int = 15) -> None:
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n  Feature importances — {target} (top {top_n}):")
    for name, score in ranked:
        bar = "#" * int(score * 200)
        print(f"    {name:<35} {score:.4f}  {bar}")


def train() -> None:
    """Main entry point: load data, train 3 models, evaluate, save artifacts."""
    city = os.getenv("CITY", "Krasnodar")
    lat = float(os.getenv("LAT", "45.03"))
    lon = float(os.getenv("LON", "38.98"))
    db_path = os.getenv("DB_PATH", "data/weather.db")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = _load_weather(db_path)
    df = build_features(raw_df)
    feature_cols = get_feature_columns(df)

    train_df, test_df = _chronological_split(df)

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    all_metrics: dict[str, dict] = {}
    models: dict[str, XGBRegressor] = {}

    for target in TARGETS:
        y_train = train_df[target]
        y_test = test_df[target]
        model, metrics = _train_one(target, X_train, y_train, X_test, y_test)
        all_metrics[target] = metrics
        models[target] = model
        _print_feature_importances(model, feature_cols, target)

    # --- save models ---
    target_to_file = {
        "temp_24h": "xgb_temp.pkl",
        "precip_24h": "xgb_precip.pkl",
        "windspeed_24h": "xgb_windspeed.pkl",
    }
    for target, filename in target_to_file.items():
        path = MODELS_DIR / filename
        joblib.dump(models[target], path)
        log.info("Saved model: %s", path)

    # --- production gate ---
    temp_mae = all_metrics["temp_24h"]["mae"]
    production_ready = bool(temp_mae < PRODUCTION_MAE_THRESHOLD)
    gate_symbol = "PASS" if production_ready else "FAIL"

    # --- save metadata ---
    metadata = {
        "trained_at": datetime.utcnow().isoformat(),
        "city": city,
        "lat": lat,
        "lon": lon,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "feature_columns": feature_cols,
        "mae_temp": all_metrics["temp_24h"]["mae"],
        "mae_precip": all_metrics["precip_24h"]["mae"],
        "mae_wind": all_metrics["windspeed_24h"]["mae"],
        "rmse_temp": all_metrics["temp_24h"]["rmse"],
        "r2_temp": all_metrics["temp_24h"]["r2"],
        "production_ready": production_ready,
    }
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved: %s", meta_path)

    # --- summary table ---
    print(f"\n{'='*60}")
    print(f"  Training complete — {city}  (lat={lat}, lon={lon})")
    print(f"{'='*60}")
    print(f"  {'Target':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'MAPE%':>8}")
    print(f"  {'-'*56}")
    labels = {"temp_24h": "Temperature (°C)", "precip_24h": "Precipitation (mm)", "windspeed_24h": "Wind speed (m/s)"}
    for target, m in all_metrics.items():
        print(f"  {labels[target]:<20} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f} {m['mape']:>7.1f}%")
    print(f"{'='*60}")
    print(f"  PRODUCTION GATE: temp MAE = {temp_mae:.3f}C  (threshold < {PRODUCTION_MAE_THRESHOLD}C)")
    print(f"  -> {gate_symbol}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
