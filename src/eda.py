"""
Exploratory Data Analysis — Pattaya weather data.

Loads weather.db and produces charts that reveal:
  1. Distribution of precipitation, wind, temperature
  2. Atmospheric precursors 1-24h before rain events
  3. Correlation matrix (all features vs targets)
  4. Pressure and humidity trends before rain / strong wind / sunny spells
  5. Monthly / seasonal precipitation patterns
  6. Feature lag correlations for precipitation target

Run:
    python src/eda.py
Output: figures saved to data/eda/ and displayed interactively.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── config ────────────────────────────────────────────────────────────────────

DB_PATH  = Path(__file__).parent.parent / "data" / "weather.db"
OUT_DIR  = Path(__file__).parent.parent / "data" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAIN_THRESHOLD  = 1.0   # mm/h — defines a "rain event"
WIND_THRESHOLD  = 8.0   # m/s  — defines "strong wind"
SUNNY_MAX_PRECIP = 0.0  # mm  total in preceding 24h

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
sns.set_palette("tab10")


# ── load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load weather table and add datetime + derived columns."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM weather ORDER BY time", conn)
    conn.close()

    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # time parts
    df["hour"]       = df.index.hour
    df["month"]      = df.index.month
    df["dayofweek"]  = df.index.dayofweek
    df["season"]     = df["month"].map(
        {12: "Winter", 1: "Winter", 2: "Winter",
          3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer",
          9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )

    # key signals
    df["pressure_change_3h"]  = df["pressure"].diff(3)
    df["pressure_change_6h"]  = df["pressure"].diff(6)
    df["pressure_change_12h"] = df["pressure"].diff(12)
    df["pressure_change_24h"] = df["pressure"].diff(24)
    df["humidity_change_3h"]  = df["humidity"].diff(3)
    df["temp_change_3h"]      = df["temp"].diff(3)
    df["temp_change_6h"]      = df["temp"].diff(6)

    df["precip_rolling_6h"]   = df["precipitation"].rolling(6).sum()
    df["precip_rolling_24h"]  = df["precipitation"].rolling(24).sum()
    df["precip_rolling_3h"]   = df["precipitation"].rolling(3).sum()

    # targets (next 24h)
    df["rain_next_24h"]  = df["precipitation"].shift(-24)
    df["wind_next_24h"]  = df["windspeed"].shift(-24)
    df["temp_next_24h"]  = df["temp"].shift(-24)

    # binary rain flag
    df["rain_event"] = (df["precipitation"] >= RAIN_THRESHOLD).astype(int)
    df["strong_wind"] = (df["windspeed"] >= WIND_THRESHOLD).astype(int)

    return df.dropna(subset=["pressure_change_24h", "rain_next_24h"])


# ── 1. overall distributions ──────────────────────────────────────────────────

def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Pattaya — Variable Distributions (2006-2026)", fontsize=14, y=1.01)

    # precipitation — log scale
    ax = axes[0, 0]
    nonzero = df["precipitation"][df["precipitation"] > 0]
    ax.hist(nonzero, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Precipitation (mm/h, non-zero only)")
    ax.set_title(f"Precipitation distribution\n(non-zero hours: {len(nonzero):,})")
    ax.set_yscale("log")

    # windspeed
    ax = axes[0, 1]
    ax.hist(df["windspeed"], bins=60, color="tomato", edgecolor="white", linewidth=0.3)
    ax.axvline(WIND_THRESHOLD, color="red", linestyle="--", label=f"{WIND_THRESHOLD} m/s threshold")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_title("Wind speed distribution")
    ax.legend(fontsize=9)

    # temperature
    ax = axes[0, 2]
    ax.hist(df["temp"], bins=60, color="orange", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Temperature (degC)")
    ax.set_title("Temperature distribution")

    # humidity
    ax = axes[1, 0]
    ax.hist(df["humidity"], bins=50, color="mediumseagreen", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Relative Humidity (%)")
    ax.set_title("Humidity distribution")

    # pressure
    ax = axes[1, 1]
    ax.hist(df["pressure"], bins=60, color="mediumpurple", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Pressure (hPa)")
    ax.set_title("Pressure distribution")

    # monthly precipitation totals
    ax = axes[1, 2]
    monthly = df.groupby("month")["precipitation"].sum() / df.index.year.nunique()
    bars = ax.bar(monthly.index, monthly.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg annual precipitation (mm)")
    ax.set_title("Average monthly precipitation")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])

    plt.tight_layout()
    out = OUT_DIR / "01_distributions.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── 2. pressure & humidity before rain events ─────────────────────────────────

def plot_precursors_rain(df: pd.DataFrame) -> None:
    """Show avg pressure, humidity, temp in the 24h before a rain event vs calm."""
    rain_hours  = df[df["rain_event"] == 1].index
    calm_hours  = df[df["rain_event"] == 0].index

    hours_before = list(range(0, 25))  # 0 = event hour, 24 = 24h prior
    precursor_vars = {
        "pressure":  ("Pressure (hPa)",          "mediumpurple"),
        "humidity":  ("Relative Humidity (%)",    "steelblue"),
        "temp":      ("Temperature (degC)",       "tomato"),
        "windspeed": ("Wind Speed (m/s)",         "darkorange"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Atmospheric Conditions Before Rain Events vs Calm\n(mean of 7205 rain-hour windows)", fontsize=13)
    axes = axes.flatten()

    for ax, (var, (ylabel, color)) in zip(axes, precursor_vars.items()):
        rain_traces  = []
        calm_traces  = []
        for h in hours_before:
            # value at time (event - h)
            shifted = df[var].shift(h)
            rain_val = shifted.loc[shifted.index.isin(rain_hours)].mean()
            calm_val = shifted.loc[shifted.index.isin(calm_hours)].mean()
            rain_traces.append(rain_val)
            calm_traces.append(calm_val)

        x = hours_before[::-1]   # 24 → 0 (left to right = approaching event)
        ax.plot(x, rain_traces[::-1],  color=color,   linewidth=2, label="Before rain")
        ax.plot(x, calm_traces[::-1],  color="gray",  linewidth=1.5, linestyle="--", label="Calm")
        ax.axvline(0, color="red", linestyle=":", linewidth=1, label="Rain hour")
        ax.set_xlabel("Hours before rain event")
        ax.set_ylabel(ylabel)
        ax.set_title(var.capitalize())
        ax.legend(fontsize=9)
        ax.invert_xaxis()

    plt.tight_layout()
    out = OUT_DIR / "02_precursors_rain.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── 3. pressure before strong wind ───────────────────────────────────────────

def plot_precursors_wind(df: pd.DataFrame) -> None:
    wind_hours = df[df["strong_wind"] == 1].index
    calm_hours = df[df["strong_wind"] == 0].index

    hours_before = list(range(0, 25))
    precursor_vars = {
        "pressure":  ("Pressure (hPa)",       "mediumpurple"),
        "temp":      ("Temperature (degC)",   "tomato"),
        "humidity":  ("Humidity (%)",         "steelblue"),
        "precipitation": ("Precipitation (mm/h)", "cornflowerblue"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Atmospheric Conditions Before Strong Wind Events (>=8 m/s) vs Calm", fontsize=13)
    axes = axes.flatten()

    for ax, (var, (ylabel, color)) in zip(axes, precursor_vars.items()):
        wind_traces = []
        calm_traces = []
        for h in hours_before:
            shifted = df[var].shift(h)
            wind_traces.append(shifted.loc[shifted.index.isin(wind_hours)].mean())
            calm_traces.append(shifted.loc[shifted.index.isin(calm_hours)].mean())

        x = hours_before[::-1]
        ax.plot(x, wind_traces[::-1], color=color,  linewidth=2, label="Before strong wind")
        ax.plot(x, calm_traces[::-1], color="gray", linewidth=1.5, linestyle="--", label="Calm")
        ax.axvline(0, color="red", linestyle=":", linewidth=1)
        ax.set_xlabel("Hours before event")
        ax.set_ylabel(ylabel)
        ax.set_title(var.capitalize())
        ax.legend(fontsize=9)
        ax.invert_xaxis()

    plt.tight_layout()
    out = OUT_DIR / "03_precursors_wind.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── 4. correlation matrix ─────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    feature_cols = [
        "temp", "humidity", "pressure", "windspeed",
        "pressure_change_3h", "pressure_change_6h", "pressure_change_12h", "pressure_change_24h",
        "humidity_change_3h", "temp_change_3h", "temp_change_6h",
        "precip_rolling_3h", "precip_rolling_6h", "precip_rolling_24h",
        "hour", "month",
    ]
    target_cols = ["rain_next_24h", "wind_next_24h", "temp_next_24h"]

    sub = df[feature_cols + target_cols].dropna()
    corr = sub.corr()[target_cols].loc[feature_cols]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7}
    )
    ax.set_title("Feature vs Target Correlations\n(targets: next-24h precip / wind / temp)", fontsize=13)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(["Precip 24h", "Wind 24h", "Temp 24h"], rotation=15)

    plt.tight_layout()
    out = OUT_DIR / "04_correlation_matrix.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()

    # print top correlations with precipitation
    print("\n=== Top correlations with rain_next_24h ===")
    print(corr["rain_next_24h"].sort_values(ascending=False).to_string())


# ── 5. lag correlations for precipitation ─────────────────────────────────────

def plot_lag_correlations(df: pd.DataFrame) -> None:
    """Correlate precipitation at each lag with the 24h-ahead value."""
    target = df["rain_next_24h"]
    lags = list(range(1, 49))

    corr_precip   = [df["precipitation"].shift(l).corr(target) for l in lags]
    corr_pressure = [df["pressure"].shift(l).corr(target) for l in lags]
    corr_humidity = [df["humidity"].shift(l).corr(target) for l in lags]
    corr_temp     = [df["temp"].shift(l).corr(target) for l in lags]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Lag Correlations with Precipitation (next 24h)", fontsize=13)

    for ax, (label, values, color) in zip(axes.flatten(), [
        ("Precipitation lag", corr_precip,   "steelblue"),
        ("Pressure lag",      corr_pressure, "mediumpurple"),
        ("Humidity lag",      corr_humidity, "mediumseagreen"),
        ("Temperature lag",   corr_temp,     "tomato"),
    ]):
        ax.bar(lags, values, color=color, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("Pearson r")
        ax.set_title(label)
        ax.set_xlim(0, 49)

    plt.tight_layout()
    out = OUT_DIR / "05_lag_correlations_precip.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── 6. seasonal rain patterns ─────────────────────────────────────────────────

def plot_seasonal_patterns(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Pattaya — Seasonal Patterns", fontsize=14)

    # hourly precipitation by season
    ax = axes[0]
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    colors = ["green", "gold", "saddlebrown", "steelblue"]
    for season, color in zip(season_order, colors):
        sub = df[df["season"] == season].groupby("hour")["precipitation"].mean()
        ax.plot(sub.index, sub.values, label=season, linewidth=2, color=color)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg precipitation (mm/h)")
    ax.set_title("Hourly precipitation by season")
    ax.legend()
    ax.set_xticks(range(0, 24, 3))

    # wind speed by hour
    ax = axes[1]
    for season, color in zip(season_order, colors):
        sub = df[df["season"] == season].groupby("hour")["windspeed"].mean()
        ax.plot(sub.index, sub.values, label=season, linewidth=2, color=color)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Avg wind speed (m/s)")
    ax.set_title("Hourly wind speed by season")
    ax.legend()
    ax.set_xticks(range(0, 24, 3))

    # pressure change before rain vs no-rain
    ax = axes[2]
    rain_mask = df["rain_event"] == 1
    for col, label, color in [
        ("pressure_change_3h",  "3h",  "purple"),
        ("pressure_change_6h",  "6h",  "blue"),
        ("pressure_change_12h", "12h", "teal"),
        ("pressure_change_24h", "24h", "navy"),
    ]:
        ax.scatter(
            df.loc[~rain_mask, col].sample(2000, random_state=42),
            df.loc[~rain_mask, "precipitation"].sample(2000, random_state=42),
            alpha=0.05, s=5, color="gray", label=None,
        )
    sub_rain = df[rain_mask].dropna(subset=["pressure_change_6h"])
    ax.scatter(sub_rain["pressure_change_6h"], sub_rain["precipitation"],
               alpha=0.3, s=8, color="red", label="Rain events")
    ax.set_xlabel("Pressure change in 6h (hPa)")
    ax.set_ylabel("Precipitation (mm/h)")
    ax.set_title("Pressure drop vs precipitation")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "06_seasonal_patterns.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── 7. pressure drop vs rain probability ──────────────────────────────────────

def plot_pressure_rain_probability(df: pd.DataFrame) -> None:
    """Bin pressure changes and compute rain probability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pressure Change vs Rain Probability", fontsize=13)

    for ax, (col, title) in zip(axes, [
        ("pressure_change_6h",  "Pressure change over 6h"),
        ("pressure_change_24h", "Pressure change over 24h"),
    ]):
        sub = df[[col, "rain_event"]].dropna()
        bins = pd.cut(sub[col], bins=20)
        prob = sub.groupby(bins, observed=True)["rain_event"].mean()
        count = sub.groupby(bins, observed=True)["rain_event"].count()
        centers = [interval.mid for interval in prob.index]

        ax2 = ax.twinx()
        ax2.bar(range(len(count)), count.values, alpha=0.2, color="gray", label="Count")
        ax2.set_ylabel("# hours in bin", color="gray")

        ax.plot(range(len(prob)), prob.values * 100, color="red", linewidth=2, marker="o", markersize=4)
        ax.set_xlabel(f"{title} bins (left=drop, right=rise)")
        ax.set_ylabel("Rain probability (%)")
        ax.set_title(title)
        tick_step = max(1, len(centers) // 8)
        ax.set_xticks(range(0, len(centers), tick_step))
        ax.set_xticklabels([f"{centers[i]:.1f}" for i in range(0, len(centers), tick_step)], fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "07_pressure_rain_probability.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── 8. summary stats printed to console ───────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    rain_hours  = (df["precipitation"] >= RAIN_THRESHOLD).sum()
    heavy_rain  = (df["precipitation"] >= 5.0).sum()
    strong_wind = (df["windspeed"] >= WIND_THRESHOLD).sum()
    zero_precip = (df["precipitation"] == 0).sum()

    print("\n" + "=" * 60)
    print("PATTAYA WEATHER EDA SUMMARY")
    print("=" * 60)
    print(f"Total hourly records : {total:>10,}")
    print(f"Date range           : {df.index.min().date()} to {df.index.max().date()}")
    print(f"Zero precipitation   : {zero_precip:>10,} ({100*zero_precip/total:.1f}%)")
    print(f"Rain hours (>=1 mm)  : {rain_hours:>10,} ({100*rain_hours/total:.1f}%)")
    print(f"Heavy rain (>=5 mm)  : {heavy_rain:>10,} ({100*heavy_rain/total:.1f}%)")
    print(f"Strong wind (>=8 m/s): {strong_wind:>10,} ({100*strong_wind/total:.1f}%)")
    print()

    # pearson correlations with next-24h precipitation
    target = df["rain_next_24h"]
    signals = {
        "humidity":            df["humidity"],
        "pressure":            df["pressure"],
        "pressure_change_3h":  df["pressure_change_3h"],
        "pressure_change_6h":  df["pressure_change_6h"],
        "pressure_change_12h": df["pressure_change_12h"],
        "precip_rolling_3h":   df["precip_rolling_3h"],
        "precip_rolling_6h":   df["precip_rolling_6h"],
        "precip_rolling_24h":  df["precip_rolling_24h"],
        "windspeed":           df["windspeed"],
        "temp":                df["temp"],
        "temp_change_3h":      df["temp_change_3h"],
    }
    print("Pearson r with rain_next_24h:")
    rows = [(name, sig.corr(target)) for name, sig in signals.items()]
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, r in rows:
        bar = "#" * int(abs(r) * 30)
        print(f"  {name:<25} {r:+.4f}  {bar}")
    print("=" * 60)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data from", DB_PATH)
    df = load_data()
    print(f"Loaded {len(df):,} rows ({df.index.min().date()} to {df.index.max().date()})")

    print_summary(df)

    print("\n[1/7] Distributions...")
    plot_distributions(df)

    print("[2/7] Rain precursors...")
    plot_precursors_rain(df)

    print("[3/7] Wind precursors...")
    plot_precursors_wind(df)

    print("[4/7] Correlation matrix...")
    plot_correlation_matrix(df)

    print("[5/7] Lag correlations...")
    plot_lag_correlations(df)

    print("[6/7] Seasonal patterns...")
    plot_seasonal_patterns(df)

    print("[7/7] Pressure-rain probability...")
    plot_pressure_rain_probability(df)

    print(f"\nAll charts saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
