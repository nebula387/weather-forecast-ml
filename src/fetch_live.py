"""Fetch live weather data from Open-Meteo Forecast API and update SQLite."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = (
    "temperature_2m,relativehumidity_2m,pressure_msl,"
    "windspeed_10m,precipitation,weathercode"
)
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5


def _get_config() -> tuple[str, float, float, str]:
    city = os.getenv("CITY", "Krasnodar")
    lat = float(os.getenv("LAT", "45.03"))
    lon = float(os.getenv("LON", "38.98"))
    db_path = os.getenv("DB_PATH", "data/weather.db")
    return city, lat, lon, db_path


def _init_tables(conn: sqlite3.Connection) -> None:
    """Create weather and forecast_raw tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            time          TEXT PRIMARY KEY,
            temp          REAL,
            humidity      REAL,
            pressure      REAL,
            windspeed     REAL,
            precipitation REAL,
            weathercode   INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_raw (
            time          TEXT PRIMARY KEY,
            temp          REAL,
            humidity      REAL,
            pressure      REAL,
            windspeed     REAL,
            precipitation REAL,
            weathercode   INTEGER
        )
    """)
    conn.commit()


def _fetch_forecast(lat: float, lon: float) -> dict:
    """Fetch past 2 days + next 2 days of hourly data from Forecast API with retries."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": 2,
        "forecast_days": 2,
        "hourly": HOURLY_VARS,
        "timezone": "Europe/Moscow",
    }
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Forecast API attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                raise


def _split_and_store(conn: sqlite3.Connection, data: dict) -> tuple[int, int]:
    """Split response into actuals/upcoming and upsert into respective tables.

    Returns:
        (actual_count, forecast_count) — rows upserted into each table.
    """
    h = data["hourly"]
    now_iso = datetime.now(timezone.utc).astimezone().replace(tzinfo=None).isoformat(timespec="minutes")

    actuals = []
    forecasts = []

    for i, ts in enumerate(h["time"]):
        row = (
            ts,
            h["temperature_2m"][i],
            h["relativehumidity_2m"][i],
            h["pressure_msl"][i],
            h["windspeed_10m"][i],
            h["precipitation"][i],
            h["weathercode"][i],
        )
        if ts <= now_iso:
            actuals.append(row)
        else:
            forecasts.append(row)

    upsert_sql = """
        INSERT OR REPLACE INTO {table}
            (time, temp, humidity, pressure, windspeed, precipitation, weathercode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    conn.executemany(upsert_sql.format(table="weather"), actuals)
    conn.executemany(upsert_sql.format(table="forecast_raw"), forecasts)
    conn.commit()

    return len(actuals), len(forecasts)


def fetch_live() -> None:
    """Main entry point: fetch live data and update both SQLite tables."""
    city, lat, lon, db_path = _get_config()
    log.info("Fetching live data for %s (lat=%.2f, lon=%.2f)", city, lat, lon)

    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Database not found: {db_path}. Run download_history.py first."
        )

    conn = sqlite3.connect(db_path)
    _init_tables(conn)

    data = _fetch_forecast(lat, lon)
    actual_count, forecast_count = _split_and_store(conn, data)
    conn.close()

    print(f"\n  Actuals upserted  -> weather table     : {actual_count} rows")
    print(f"  Forecast stored   -> forecast_raw table : {forecast_count} rows")
    print(f"  Forecast horizon  : ~{forecast_count} hours ahead\n")


if __name__ == "__main__":
    fetch_live()
