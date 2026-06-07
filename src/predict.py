"""Load live data + trained models, produce 24h forecast for all targets."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv

from features import TARGETS, build_features, get_feature_columns
import models  # noqa: F401 — needed so joblib can deserialize TwoStagePrecipModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
OUTPUT_PATH = Path("data/processed/latest_predictions.csv")

MODEL_FILES = {
    "temp_24h": "xgb_temp.pkl",
    "precip_24h": "xgb_precip.pkl",
    "windspeed_24h": "xgb_windspeed.pkl",
}

# need enough history for lag-24 warmup + rolling-168 window + buffer
HISTORY_ROWS = 250


def _load_metadata() -> dict:
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}. Run train.py first."
        )
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _load_models() -> dict:
    """Load all three XGBoost models from disk."""
    models = {}
    for target, filename in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
        models[target] = joblib.load(path)
        log.info("Loaded model: %s", path)
    return models


def _load_merged_data(db_path: str) -> tuple[pd.DataFrame, str]:
    """Load historical weather (last HISTORY_ROWS) + forecast_raw, merged by time.

    Returns:
        (merged_df, last_actual_time) where last_actual_time is the ISO timestamp
        of the most recent row in the weather table.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}. Run download_history.py first.")

    conn = sqlite3.connect(db_path)

    try:
        weather = pd.read_sql(
            f"SELECT * FROM weather ORDER BY time DESC LIMIT {HISTORY_ROWS}",
            conn,
        )
    except Exception as exc:
        conn.close()
        raise RuntimeError("weather table missing or empty") from exc

    last_actual_time = weather["time"].max()

    try:
        forecast_raw = pd.read_sql("SELECT * FROM forecast_raw ORDER BY time", conn)
    except Exception as exc:
        conn.close()
        raise RuntimeError("forecast_raw table missing. Run fetch_live.py first.") from exc

    conn.close()

    merged = pd.concat([weather, forecast_raw], ignore_index=True)
    merged = merged.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    log.info(
        "Merged data: %d rows  (weather=%d, forecast_raw=%d)",
        len(merged), len(weather), len(forecast_raw),
    )
    return merged, last_actual_time


def predict() -> pd.DataFrame:
    """Main entry point: run inference and return/save 24-row predictions DataFrame."""
    db_path = os.getenv("DB_PATH", "data/weather.db")
    city = os.getenv("CITY", "Krasnodar")

    metadata = _load_metadata()
    models = _load_models()
    feature_cols: list[str] = metadata["feature_columns"]

    raw_df, last_actual_time = _load_merged_data(db_path)

    # build features in inference mode: no targets, no NaN-drop on targets
    df = build_features(raw_df, inference=True)

    # forecast rows = rows where time > last actual weather timestamp
    df["time"] = pd.to_datetime(df["time"])
    last_actual_dt = pd.to_datetime(last_actual_time)
    forecast_df = df[df["time"] > last_actual_dt].head(24)

    if len(forecast_df) == 0:
        log.warning(
            "No forecast rows found after %s. Run fetch_live.py to update forecast_raw.",
            last_actual_time,
        )

    if len(forecast_df) < 24:
        log.warning("Only %d forecast rows available (expected 24)", len(forecast_df))

    X_forecast = forecast_df[feature_cols]

    results = pd.DataFrame()
    results["timestamp"] = forecast_df["time"].values

    for target, model in models.items():
        results[target] = model.predict(X_forecast)

    results = results[["timestamp", "temp_24h", "precip_24h", "windspeed_24h"]].copy()
    results.columns = ["timestamp", "pred_temp", "pred_precip", "pred_windspeed"]

    # confidence bounds on temperature (±MAE)
    mae_temp = metadata.get("mae_temp", 0.0)
    results["temp_lower"] = results["pred_temp"] - mae_temp
    results["temp_upper"] = results["pred_temp"] + mae_temp

    # round for readability
    results["pred_temp"] = results["pred_temp"].round(1)
    results["pred_precip"] = results["pred_precip"].clip(lower=0).round(2)
    results["pred_windspeed"] = results["pred_windspeed"].clip(lower=0).round(1)
    results["temp_lower"] = results["temp_lower"].round(1)
    results["temp_upper"] = results["temp_upper"].round(1)

    results = results.reset_index(drop=True)

    # save CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    log.info("Predictions saved: %s", OUTPUT_PATH)

    # print table
    print(f"\n  24-hour forecast — {city}")
    print(f"  {'Timestamp':<20} {'Temp C':>8} {'[Low,High]':>14} {'Precip mm':>10} {'Wind m/s':>9}")
    print(f"  {'-'*65}")
    for _, row in results.iterrows():
        ts = str(row["timestamp"])[:16]
        bounds = f"[{row['temp_lower']:.1f},{row['temp_upper']:.1f}]"
        print(
            f"  {ts:<20} {row['pred_temp']:>8.1f} {bounds:>14}"
            f" {row['pred_precip']:>10.2f} {row['pred_windspeed']:>9.1f}"
        )
    print()

    return results


if __name__ == "__main__":
    predict()
