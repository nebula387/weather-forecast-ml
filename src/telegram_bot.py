"""Daily Telegram forecast sender — fetch live data, predict, send message."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# run from weather-forecast-ml/ regardless of where the script is called from
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "latest_predictions.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _run_pipeline() -> None:
    """Fetch fresh live data and regenerate predictions before sending."""
    python = sys.executable
    for script in ["src/fetch_live.py", "src/predict.py"]:
        log.info("Running %s ...", script)
        result = subprocess.run(
            [python, script],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{script} failed:\n{result.stderr}"
            )
        log.info("%s done", script)


def _load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}. Run predict.py first."
        )
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])


def _load_mae() -> float:
    if not METADATA_PATH.exists():
        return 0.0
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("mae_temp", 0.0)


_PERIODS = [
    ("Morning",   "🌅",  6, 12),
    ("Afternoon", "🌞", 12, 18),
    ("Evening",   "🌆", 18, 24),
    ("Night",     "🌙",  0,  6),  # next day's early hours — always last
]


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


def _build_message(df: pd.DataFrame, mae_temp: float) -> str:
    """Build a structured daily forecast message broken down by time of day."""
    city = os.getenv("CITY", "Krasnodar")
    today = date.today().strftime("%A, %B %d")

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour

    # day-level summary
    day_min = df["pred_temp"].min()
    day_max = df["pred_temp"].max()
    day_precip = df["pred_precip"].sum()
    day_wind = df["pred_windspeed"].max()

    header_icon = _period_icon(day_precip, day_max)
    lines = [
        f"{header_icon} <b>{city} — {today}</b>",
        "",
    ]

    for label, emoji, h_start, h_end in _PERIODS:
        if h_start < h_end:
            mask = (df["hour"] >= h_start) & (df["hour"] < h_end)
        else:
            mask = (df["hour"] >= h_start) | (df["hour"] < h_end)

        seg = df[mask]
        if seg.empty:
            continue

        t_min = seg["pred_temp"].min()
        t_max = seg["pred_temp"].max()
        precip = seg["pred_precip"].sum()
        wind = seg["pred_windspeed"].max()
        cond = _period_icon(precip, t_max)

        time_label = (
            f"{h_start:02d}:00–{h_end:02d}:00"
            if h_end != 24
            else f"{h_start:02d}:00–00:00"
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


def _send_message(text: str) -> None:
    """Send a message via Telegram Bot API."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or token == "your_token_here":
        raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env")
    if not chat_id:
        raise ValueError("TELEGRAM_CHAT_ID is not set in .env")

    url = TELEGRAM_API.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    resp = requests.post(url, json=payload, timeout=15)
    if not resp.ok:
        raise RuntimeError(f"Telegram API {resp.status_code}: {resp.text}")
    result = resp.json()

    if not result.get("ok"):
        raise RuntimeError(f"Telegram API error: {result}")

    log.info("Message sent. message_id=%s", result["result"]["message_id"])


def send_daily_forecast() -> None:
    """Main entry point: update data, build forecast message, send to Telegram."""
    log.info("=== Daily forecast pipeline started ===")

    _run_pipeline()

    df = _load_predictions()
    mae_temp = _load_mae()
    message = _build_message(df, mae_temp)

    log.info("Sending forecast to Telegram ...")
    _send_message(message)

    clean = (
        message
        .replace("<b>", "").replace("</b>", "")
        .replace("<i>", "").replace("</i>", "")
        .encode("ascii", errors="replace").decode("ascii")
    )
    print("\n--- Message sent ---")
    print(clean)
    print("--------------------\n")


if __name__ == "__main__":
    send_daily_forecast()
