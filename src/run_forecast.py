"""Standalone scheduled forecast for GitHub Actions — no SQLite, no .env needed."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, get_feature_columns
import models  # noqa: F401 — needed so joblib can deserialize TwoStagePrecipModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MODELS_DIR = Path(__file__).parent.parent / "models"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

_PERIODS = [
    ("Morning",   "🌅",  6, 12),
    ("Afternoon", "🌞", 12, 18),
    ("Evening",   "🌆", 18, 24),
    ("Night",     "🌙",  0,  6),
]


# ── data ─────────────────────────────────────────────────────────────────────

def fetch_data(lat: float, lon: float, past_days: int = 8) -> pd.DataFrame:
    """Fetch past_days of actuals + 2 days forecast from Forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 2,
        "hourly": (
            "temperature_2m,relativehumidity_2m,pressure_msl,"
            "windspeed_10m,precipitation,weathercode"
        ),
        "timezone": "Europe/Moscow",
    }
    for attempt in range(1, 4):
        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=30)
            resp.raise_for_status()
            h = resp.json()["hourly"]
            return pd.DataFrame({
                "time":          h["time"],
                "temp":          h["temperature_2m"],
                "humidity":      h["relativehumidity_2m"],
                "pressure":      h["pressure_msl"],
                "windspeed":     h["windspeed_10m"],
                "precipitation": h["precipitation"],
                "weathercode":   h["weathercode"],
            })
        except requests.RequestException as exc:
            log.warning("API attempt %d/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(5 * attempt)
            else:
                raise


# ── prediction ────────────────────────────────────────────────────────────────

def run_prediction(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Build features, load models, return (predictions_df, mae_temp)."""
    df = build_features(raw_df, inference=True)
    feature_cols = get_feature_columns(df)

    df["time"] = pd.to_datetime(df["time"])
    now = pd.Timestamp.now(tz="Europe/Moscow").tz_localize(None)
    forecast_rows = df[df["time"] > now].head(24)

    if forecast_rows.empty:
        raise RuntimeError("No future rows found — check past_days setting.")

    model_files = {
        "temp_24h":      "xgb_temp.pkl",
        "precip_24h":    "xgb_precip.pkl",
        "windspeed_24h": "xgb_windspeed.pkl",
    }
    X = forecast_rows[feature_cols]
    results = pd.DataFrame({"timestamp": forecast_rows["time"].values})

    for target, fname in model_files.items():
        model = joblib.load(MODELS_DIR / fname)
        results[target] = model.predict(X)

    results.columns = ["timestamp", "pred_temp", "pred_precip", "pred_windspeed"]
    results["pred_precip"]    = results["pred_precip"].clip(lower=0).round(2)
    results["pred_windspeed"] = results["pred_windspeed"].clip(lower=0).round(1)
    results["pred_temp"]      = results["pred_temp"].round(1)

    meta_path = MODELS_DIR / "metadata.json"
    mae_temp = 0.0
    if meta_path.exists():
        mae_temp = json.loads(meta_path.read_text()).get("mae_temp", 0.0)

    return results.reset_index(drop=True), mae_temp


# ── message ───────────────────────────────────────────────────────────────────

def _period_icon(precip: float, temp_max: float) -> str:
    if precip > 3:
        return "🌧"
    if precip > 0.5:
        return "🌦"
    if temp_max > 28:
        return "☀"
    if temp_max > 18:
        return "⛅"
    return "🌤"


def build_message(df: pd.DataFrame, mae_temp: float) -> str:
    """Build structured Telegram HTML message broken down by time of day."""
    city = os.getenv("CITY", "Krasnodar")
    today = date.today().strftime("%A, %B %d")

    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    day_min    = df["pred_temp"].min()
    day_max    = df["pred_temp"].max()
    day_precip = df["pred_precip"].sum()
    day_wind   = df["pred_windspeed"].max()

    header_icon = _period_icon(day_precip, day_max)
    lines = [f"{header_icon} <b>{city} — {today}</b>", ""]

    for label, emoji, h_start, h_end in _PERIODS:
        if h_start < h_end:
            mask = (df["hour"] >= h_start) & (df["hour"] < h_end)
        else:
            mask = (df["hour"] >= h_start) | (df["hour"] < h_end)

        seg = df[mask]
        if seg.empty:
            continue

        t_min  = seg["pred_temp"].min()
        t_max  = seg["pred_temp"].max()
        precip = seg["pred_precip"].sum()
        wind   = seg["pred_windspeed"].max()
        cond   = _period_icon(precip, t_max)
        time_label = (
            f"{h_start:02d}:00–{h_end:02d}:00"
            if h_end != 24 else f"{h_start:02d}:00–00:00"
        )

        lines += [
            f"{emoji} <b>{label}</b>  <i>{time_label}</i>  {cond}",
            f"   🌡 {t_min:.0f}°C – {t_max:.0f}°C"
            f"   🌧 {precip:.1f} mm"
            f"   💨 {wind:.0f} m/s",
            "",
        ]

    lines += [
        "─────────────────────",
        f"📊 <b>Day:</b> {day_min:.0f}°C – {day_max:.0f}°C"
        f"  |  {day_precip:.1f} mm  |  up to {day_wind:.0f} m/s",
        f"<i>Model accuracy ±{mae_temp:.2f}°C</i>",
    ]
    return "\n".join(lines)


# ── telegram ──────────────────────────────────────────────────────────────────

def send_telegram(message: str) -> None:
    """Send HTML message via Telegram Bot API."""
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")

    resp = requests.post(
        TELEGRAM_API.format(token=token),
        json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
        timeout=15,
    )
    if not resp.ok:
        raise RuntimeError(f"Telegram API {resp.status_code}: {resp.text}")
    log.info("Sent. message_id=%s", resp.json()["result"]["message_id"])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    city = os.getenv("CITY", "Krasnodar")
    lat  = float(os.getenv("LAT", "45.03"))
    lon  = float(os.getenv("LON", "38.98"))

    log.info("=== Scheduled forecast: %s ===", city)

    log.info("Fetching live data ...")
    raw_df = fetch_data(lat, lon)

    log.info("Running prediction ...")
    predictions, mae_temp = run_prediction(raw_df)
    log.info("Got %d forecast rows. mae_temp=%.3f", len(predictions), mae_temp)

    message = build_message(predictions, mae_temp)

    log.info("Sending to Telegram ...")
    send_telegram(message)

    log.info("Done.")


if __name__ == "__main__":
    main()
