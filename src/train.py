"""Train one XGBoost model per target, evaluate, and save with metadata.

Precipitation uses a two-stage approach to handle the extreme class imbalance
(93% of hours have zero precipitation):
  Stage 1 — XGBClassifier: predicts P(rain > 0.5 mm in next 24h)
  Stage 2 — XGBRegressor : predicts amount, trained only on rainy hours
  Combined: output = amount if P(rain) > 0.50 else 0

clf_threshold=0.50 (raised from an initial 0.30 on 2026-09-05): at 0.30 the
classifier fired "rain" on 40.7% of test-set hours vs an actual base rate of
6.8%, i.e. only ~1 in 8 "rain" calls was real (precision 0.127). Precision
plateaus around 0.15 for any threshold >= 0.40, so 0.50 was chosen as the best
recall obtainable at that plateau (~0.47) rather than sacrificing more recall
for zero further precision gain at higher thresholds.
"""

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
from models import RAIN_THRESHOLD, TwoStagePrecipModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PRODUCTION_MAE_THRESHOLD = 2.0  # degrees C

XGB_BASE = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_weather(db_path: str) -> pd.DataFrame:
    """Load full weather table from SQLite."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM weather ORDER BY time", conn)
    conn.close()
    log.info("Loaded %d rows from weather table", len(df))
    return df


def _chronological_split(
    df: pd.DataFrame, test_days: int = 365
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split preserving time order. Test = last test_days days."""
    df["time"] = pd.to_datetime(df["time"])
    cutoff = df["time"].max() - pd.Timedelta(days=test_days)
    train  = df[df["time"] <= cutoff].copy()
    test   = df[df["time"] >  cutoff].copy()
    log.info("Train: %d rows  Test: %d rows  Cutoff: %s", len(train), len(test), cutoff.date())
    return train, test


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _train_standard(
    target: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
) -> tuple[XGBRegressor, dict]:
    """Train one standard XGBoost regressor and return (model, metrics)."""
    log.info("Training model for target: %s", target)
    model = XGBRegressor(**XGB_BASE, eval_metric="mae")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))
    mape = _mape(y_test.values, y_pred)

    metrics = {
        "mae":  float(round(mae,  4)),
        "rmse": float(round(rmse, 4)),
        "r2":   float(round(r2,   4)),
        "mape": float(round(mape, 2)),
    }
    log.info("%s  MAE=%.3f  RMSE=%.3f  R2=%.3f", target, mae, rmse, r2)
    return model, metrics


def _eval_precip(model: TwoStagePrecipModel, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute metrics for two-stage precipitation model."""
    y_pred = model.predict(X_test)
    mae  = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = float(r2_score(y_test, y_pred))
    mape = _mape(y_test.values, y_pred)

    # Classifier AUC-style metric
    y_bin  = (y_test > RAIN_THRESHOLD).astype(int)
    p_rain = model.clf.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_bin, p_rain)) if y_bin.sum() > 0 else float("nan")

    log.info("precip_24h  MAE=%.3f  RMSE=%.3f  R2=%.3f  AUC=%.3f", mae, rmse, r2, auc)
    return {
        "mae":  float(round(mae,  4)),
        "rmse": float(round(rmse, 4)),
        "r2":   float(round(r2,   4)),
        "mape": float(round(mape, 2)),
        "auc":  float(round(auc,  4)),
    }


def _print_feature_importances(
    model: object, feature_cols: list[str], target: str, top_n: int = 15
) -> None:
    importances = model.feature_importances_
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"\n  Feature importances -- {target} (top {top_n}):")
    for name, score in ranked:
        bar = "#" * int(score * 200)
        print(f"    {name:<40} {score:.4f}  {bar}")


# ── main ─────────────────────────────────────────────────────────────────────

def train() -> None:
    """Load data, train 3 models (incl. two-stage precip), evaluate, save artifacts."""
    city    = os.getenv("CITY",    "Pattaya")
    lat     = float(os.getenv("LAT",  "12.92"))
    lon     = float(os.getenv("LON",  "100.88"))
    db_path = os.getenv("DB_PATH", "data/weather.db")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df      = _load_weather(db_path)
    df          = build_features(raw_df)
    feature_cols = get_feature_columns(df)
    log.info("Feature columns: %d", len(feature_cols))

    train_df, test_df = _chronological_split(df)

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    all_metrics: dict[str, dict] = {}
    models: dict[str, object]    = {}

    # --- temperature ---
    model_temp, m_temp = _train_standard(
        "temp_24h", X_train, train_df["temp_24h"], X_test, test_df["temp_24h"]
    )
    all_metrics["temp_24h"] = m_temp
    models["temp_24h"] = model_temp
    _print_feature_importances(model_temp, feature_cols, "temp_24h")

    # --- precipitation (two-stage) ---
    log.info("Training two-stage precipitation model ...")
    precip_model = TwoStagePrecipModel(clf_threshold=0.50)
    precip_model.fit(X_train, train_df["precip_24h"], X_test, test_df["precip_24h"])
    m_precip = _eval_precip(precip_model, X_test, test_df["precip_24h"])
    all_metrics["precip_24h"] = m_precip
    models["precip_24h"] = precip_model
    _print_feature_importances(precip_model, feature_cols, "precip_24h")

    # --- wind speed ---
    model_wind, m_wind = _train_standard(
        "windspeed_24h", X_train, train_df["windspeed_24h"], X_test, test_df["windspeed_24h"]
    )
    all_metrics["windspeed_24h"] = m_wind
    models["windspeed_24h"] = model_wind
    _print_feature_importances(model_wind, feature_cols, "windspeed_24h")

    # --- save models ---
    target_to_file = {
        "temp_24h":      "xgb_temp.pkl",
        "precip_24h":    "xgb_precip.pkl",
        "windspeed_24h": "xgb_windspeed.pkl",
    }
    for target, filename in target_to_file.items():
        path = MODELS_DIR / filename
        joblib.dump(models[target], path)
        log.info("Saved: %s", path)

    # --- production gate ---
    temp_mae        = all_metrics["temp_24h"]["mae"]
    production_ready = bool(temp_mae < PRODUCTION_MAE_THRESHOLD)
    gate_symbol     = "PASS" if production_ready else "FAIL"

    # --- save metadata ---
    metadata = {
        "trained_at":       datetime.utcnow().isoformat(),
        "city":             city,
        "lat":              lat,
        "lon":              lon,
        "train_rows":       len(train_df),
        "test_rows":        len(test_df),
        "feature_count":    len(feature_cols),
        "feature_columns":  feature_cols,
        "mae_temp":         all_metrics["temp_24h"]["mae"],
        "mae_precip":       all_metrics["precip_24h"]["mae"],
        "mae_wind":         all_metrics["windspeed_24h"]["mae"],
        "rmse_temp":        all_metrics["temp_24h"]["rmse"],
        "r2_temp":          all_metrics["temp_24h"]["r2"],
        "r2_precip":        all_metrics["precip_24h"]["r2"],
        "auc_precip":       all_metrics["precip_24h"].get("auc", 0.0),
        "production_ready": production_ready,
        "precip_model":     "TwoStage(clf+reg)",
    }
    meta_path = MODELS_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Metadata saved: %s", meta_path)

    # --- summary table ---
    print(f"\n{'='*65}")
    print(f"  Training complete -- {city}  (lat={lat}, lon={lon})")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"{'='*65}")
    print(f"  {'Target':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Extra':>10}")
    print(f"  {'-'*60}")
    labels = {
        "temp_24h":      "Temperature (C)",
        "precip_24h":    "Precipitation (mm)",
        "windspeed_24h": "Wind speed (m/s)",
    }
    for target, m in all_metrics.items():
        extra = f"AUC={m['auc']:.3f}" if "auc" in m else f"MAPE={m['mape']:.1f}%"
        print(f"  {labels[target]:<22} {m['mae']:>8.3f} {m['rmse']:>8.3f} {m['r2']:>8.3f} {extra:>10}")
    print(f"{'='*65}")
    print(f"  PRODUCTION GATE: temp MAE = {temp_mae:.3f}C  (threshold < {PRODUCTION_MAE_THRESHOLD}C)")
    print(f"  -> {gate_symbol}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    train()
