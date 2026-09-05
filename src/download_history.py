"""Fetch 20 years of hourly weather data from Open-Meteo Archive API and store in SQLite."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = (
    "temperature_2m,relativehumidity_2m,pressure_msl,"
    "windspeed_10m,precipitation,weathercode,"
    "cloudcover,dew_point_2m,shortwave_radiation,vapour_pressure_deficit"
)
RAW_DIR = Path("data/raw")
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 5  # seconds


def _get_config() -> tuple[str, float, float, str]:
    city = os.getenv("CITY", "Pattaya")
    lat = float(os.getenv("LAT", "12.92"))
    lon = float(os.getenv("LON", "100.88"))
    db_path = os.getenv("DB_PATH", "data/weather.db")
    return city, lat, lon, db_path


def _init_db(db_path: str) -> sqlite3.Connection:
    """Create SQLite DB and weather table if they don't exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather (
            time                    TEXT PRIMARY KEY,
            temp                    REAL,
            humidity                REAL,
            pressure                REAL,
            windspeed               REAL,
            precipitation           REAL,
            weathercode             INTEGER,
            cloudcover              REAL,
            dew_point               REAL,
            shortwave_radiation     REAL,
            vapour_pressure_deficit REAL
        )
    """)
    conn.commit()
    return conn


def _fetch_year(year: int, lat: float, lon: float) -> dict:
    """Fetch one year of hourly data from Archive API with retry logic."""
    start = f"{year}-01-01"
    end = f"{year}-12-31" if year < date.today().year else str(date.today() - timedelta(days=1))

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "Asia/Bangkok",
    }

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Year %d — attempt %d/%d failed: %s", year, attempt, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF * attempt)
            else:
                raise


def _upsert_rows(conn: sqlite3.Connection, data: dict) -> int:
    """Parse API response and upsert rows into weather table. Returns inserted count."""
    h = data["hourly"]
    rows = list(zip(
        h["time"],
        h["temperature_2m"],
        h["relativehumidity_2m"],
        h["pressure_msl"],
        h["windspeed_10m"],
        h["precipitation"],
        h["weathercode"],
        h["cloudcover"],
        h["dew_point_2m"],
        h["shortwave_radiation"],
        h["vapour_pressure_deficit"],
    ))
    conn.executemany(
        """
        INSERT OR REPLACE INTO weather
            (time, temp, humidity, pressure, windspeed, precipitation, weathercode,
             cloudcover, dew_point, shortwave_radiation, vapour_pressure_deficit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _save_raw(year: int, data: dict) -> None:
    """Save raw API JSON response to data/raw/history_{year}.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"history_{year}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    log.debug("Raw JSON saved: %s", path)


def _check_missing_hours(conn: sqlite3.Connection) -> int:
    """Return count of expected hours missing from the weather table."""
    row = conn.execute("SELECT MIN(time), MAX(time) FROM weather").fetchone()
    if not row[0]:
        return 0
    start_dt = datetime.fromisoformat(row[0])
    end_dt = datetime.fromisoformat(row[1])
    expected = int((end_dt - start_dt).total_seconds() / 3600) + 1
    actual = conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
    return expected - actual


def download_history() -> None:
    """Main entry point: fetch 20 years of data and store in SQLite."""
    city, lat, lon, db_path = _get_config()
    log.info("City: %s  lat=%.2f  lon=%.2f", city, lat, lon)

    conn = _init_db(db_path)

    today = date.today()
    start_year = today.year - 20
    end_year = today.year

    total_inserted = 0
    years = list(range(start_year, end_year + 1))

    for year in tqdm(years, desc="Fetching years", unit="yr"):
        raw_path = RAW_DIR / f"history_{year}.json"

        if raw_path.exists():
            log.debug("Year %d: using cached raw JSON", year)
            with open(raw_path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            log.debug("Year %d: fetching from API", year)
            data = _fetch_year(year, lat, lon)
            _save_raw(year, data)

        inserted = _upsert_rows(conn, data)
        total_inserted += inserted
        log.debug("Year %d: %d rows upserted", year, inserted)

    row = conn.execute("SELECT MIN(time), MAX(time), COUNT(*) FROM weather").fetchone()
    missing = _check_missing_hours(conn)
    conn.close()

    print(f"\n{'='*50}")
    print(f"  City          : {city}")
    print(f"  Total rows    : {row[2]:,}")
    print(f"  Date range    : {row[0]} to {row[1]}")
    print(f"  Missing hours : {missing}")
    print(f"  DB path       : {db_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    download_history()
