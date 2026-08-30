"""Generate notebooks/eda_pattaya.ipynb from cell definitions."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell

cells = [

M("""\
# Pattaya Weather EDA

Exploratory analysis of 20 years of hourly data to find atmospheric precursors for:
- **Rain events** (precipitation >= 1 mm/h)
- **Strong wind** (>= 8 m/s)
- **Sunny weather**

Key findings feed directly into `features.py` to improve the precipitation model (baseline R²=0.002).
"""),

C("""\
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
sns.set_palette("tab10")

DB_PATH = Path("..") / "data" / "weather.db"
RAIN_THRESHOLD = 1.0    # mm/h
WIND_THRESHOLD = 8.0    # m/s
OUT_DIR = Path("..") / "data" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)
"""),

M("## 1. Load data"),

C("""\
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM weather ORDER BY time", conn)
conn.close()

df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time").sort_index()

# Time features
df["hour"]   = df.index.hour
df["month"]  = df.index.month
df["season"] = df["month"].map({
    12:"Winter",1:"Winter",2:"Winter",
    3:"Spring",4:"Spring",5:"Spring",
    6:"Summer",7:"Summer",8:"Summer",
    9:"Autumn",10:"Autumn",11:"Autumn"
})

# Atmospheric change signals
for h in [3, 6, 12, 24]:
    df[f"pressure_change_{h}h"] = df["pressure"].diff(h)
    df[f"humidity_change_{h}h"] = df["humidity"].diff(h)

df["temp_change_3h"] = df["temp"].diff(3)
df["temp_change_6h"] = df["temp"].diff(6)

# Rolling precipitation sums
for w in [3, 6, 24]:
    df[f"precip_rolling_{w}h"] = df["precipitation"].rolling(w).sum()

# Targets: next 24h
df["rain_next_24h"] = df["precipitation"].shift(-24)
df["wind_next_24h"] = df["windspeed"].shift(-24)
df["temp_next_24h"] = df["temp"].shift(-24)

# Binary event flags
df["rain_event"]  = (df["precipitation"] >= RAIN_THRESHOLD).astype(int)
df["strong_wind"] = (df["windspeed"] >= WIND_THRESHOLD).astype(int)

df = df.dropna(subset=["pressure_change_24h", "rain_next_24h"])
print(f"Loaded {len(df):,} rows  |  {df.index.min().date()} to {df.index.max().date()}")
df.describe().T.round(2)
"""),

M("## 2. Class imbalance — the root cause of poor precipitation model"),

C("""\
total        = len(df)
zero_precip  = (df["precipitation"] == 0).sum()
rain_hours   = (df["precipitation"] >= RAIN_THRESHOLD).sum()
heavy_rain   = (df["precipitation"] >= 5.0).sum()
strong_wind  = (df["windspeed"] >= WIND_THRESHOLD).sum()

print("=" * 52)
print("  PATTAYA — CLASS IMBALANCE SUMMARY")
print("=" * 52)
print(f"  Total rows           : {total:>10,}")
print(f"  Zero precipitation   : {zero_precip:>10,}  ({100*zero_precip/total:.1f}%)")
print(f"  Rain rows (>=1 mm/h) : {rain_hours:>10,}  ({100*rain_hours/total:.1f}%)")
print(f"  Heavy rain (>=5 mm/h): {heavy_rain:>10,}  ({100*heavy_rain/total:.1f}%)")
print(f"  Strong wind (>=8 m/s): {strong_wind:>10,}  ({100*strong_wind/total:.1f}%)")
print()
print("  => 87% zeros explain why regression predicts ~0 (R2=0.002)")
print("  => Fix: two-stage model (XGBClassifier + XGBRegressor)")
"""),

M("## 3. Distributions"),

C("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle("Pattaya — Variable Distributions (2006-2026)", fontsize=14)

nonzero = df["precipitation"][df["precipitation"] > 0]
axes[0,0].hist(nonzero, bins=60, color="steelblue", edgecolor="white", lw=0.3)
axes[0,0].set_yscale("log")
axes[0,0].set_title(f"Precipitation (non-zero: {len(nonzero):,})")
axes[0,0].set_xlabel("mm/h")

axes[0,1].hist(df["windspeed"], bins=60, color="tomato", edgecolor="white", lw=0.3)
axes[0,1].axvline(WIND_THRESHOLD, color="red", ls="--", label=f"{WIND_THRESHOLD} m/s")
axes[0,1].set_title("Wind speed"); axes[0,1].set_xlabel("m/s"); axes[0,1].legend()

axes[0,2].hist(df["temp"], bins=60, color="orange", edgecolor="white", lw=0.3)
axes[0,2].set_title("Temperature"); axes[0,2].set_xlabel("degC")

axes[1,0].hist(df["humidity"], bins=50, color="mediumseagreen", edgecolor="white", lw=0.3)
axes[1,0].set_title("Humidity"); axes[1,0].set_xlabel("%")

axes[1,1].hist(df["pressure"], bins=60, color="mediumpurple", edgecolor="white", lw=0.3)
axes[1,1].set_title("Pressure"); axes[1,1].set_xlabel("hPa")

monthly = df.groupby("month")["precipitation"].sum() / df.index.year.nunique()
axes[1,2].bar(monthly.index, monthly.values, color="steelblue")
axes[1,2].set_title("Avg monthly precip (mm)")
axes[1,2].set_xlabel("Month")
axes[1,2].set_xticks(range(1,13))
axes[1,2].set_xticklabels(list("JFMAMJJASOND"))

plt.tight_layout()
plt.savefig(OUT_DIR / "01_distributions.png", bbox_inches="tight")
plt.show()
"""),

M("## 4. Atmospheric precursors before rain events"),

C("""\
rain_idx = df[df["rain_event"] == 1].index
calm_idx = df[df["rain_event"] == 0].index
hours_before = list(range(0, 25))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Atmospheric Conditions 24h Before Rain vs Calm", fontsize=13)

for ax, (var, ylabel, color) in zip(axes.flatten(), [
    ("pressure",  "Pressure (hPa)",     "mediumpurple"),
    ("humidity",  "Humidity (%)",        "steelblue"),
    ("temp",      "Temperature (degC)", "tomato"),
    ("windspeed", "Wind speed (m/s)",   "darkorange"),
]):
    rain_t, calm_t = [], []
    for h in hours_before:
        s = df[var].shift(h)
        rain_t.append(s.loc[s.index.isin(rain_idx)].mean())
        calm_t.append(s.loc[s.index.isin(calm_idx)].mean())
    x = hours_before[::-1]
    ax.plot(x, rain_t[::-1], color=color, lw=2, label="Before rain")
    ax.plot(x, calm_t[::-1], color="gray", lw=1.5, ls="--", label="Calm")
    ax.axvline(0, color="red", ls=":", lw=1, label="Rain hour")
    ax.invert_xaxis()
    ax.set_xlabel("Hours before rain event")
    ax.set_ylabel(ylabel)
    ax.set_title(var)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "02_precursors_rain.png", bbox_inches="tight")
plt.show()
"""),

M("## 5. Correlation matrix — features vs targets"),

C("""\
feat_cols = [
    "temp", "humidity", "pressure", "windspeed",
    "pressure_change_3h", "pressure_change_6h", "pressure_change_12h", "pressure_change_24h",
    "humidity_change_3h", "temp_change_3h", "temp_change_6h",
    "precip_rolling_3h", "precip_rolling_6h", "precip_rolling_24h",
    "hour", "month",
]
tgt_cols = ["rain_next_24h", "wind_next_24h", "temp_next_24h"]

sub  = df[feat_cols + tgt_cols].dropna()
corr = sub.corr()[tgt_cols].loc[feat_cols]

fig, ax = plt.subplots(figsize=(9, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Feature vs Target Correlations", fontsize=13)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.set_xticklabels(["Precip 24h", "Wind 24h", "Temp 24h"], rotation=15)
plt.tight_layout()
plt.savefig(OUT_DIR / "04_correlation_matrix.png", bbox_inches="tight")
plt.show()

print("Top correlations with rain_next_24h:")
print(corr["rain_next_24h"].sort_values(ascending=False).to_string())
"""),

M("## 6. Lag correlations with precipitation next 24h"),

C("""\
target = df["rain_next_24h"]
lags   = list(range(1, 49))

corr_precip   = [df["precipitation"].shift(l).corr(target) for l in lags]
corr_pressure = [df["pressure"].shift(l).corr(target) for l in lags]
corr_humidity = [df["humidity"].shift(l).corr(target) for l in lags]
corr_temp     = [df["temp"].shift(l).corr(target) for l in lags]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Lag Correlations with Precipitation (next 24h)", fontsize=13)

for ax, (label, values, color) in zip(axes.flatten(), [
    ("Precipitation", corr_precip,   "steelblue"),
    ("Pressure",      corr_pressure, "mediumpurple"),
    ("Humidity",      corr_humidity, "mediumseagreen"),
    ("Temperature",   corr_temp,     "tomato"),
]):
    ax.bar(lags, values, color=color, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Pearson r")
    ax.set_title(label)
    ax.set_xlim(0, 49)

plt.tight_layout()
plt.savefig(OUT_DIR / "05_lag_correlations.png", bbox_inches="tight")
plt.show()
"""),

M("## 7. Pressure drop vs rain probability"),

C("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Pressure Change vs Rain Probability", fontsize=13)

for ax, (col, title) in zip(axes, [
    ("pressure_change_6h",  "Pressure change over 6h"),
    ("pressure_change_24h", "Pressure change over 24h"),
]):
    sub    = df[[col, "rain_event"]].dropna()
    bins   = pd.cut(sub[col], bins=20)
    prob   = sub.groupby(bins, observed=True)["rain_event"].mean()
    count  = sub.groupby(bins, observed=True)["rain_event"].count()
    centers = [i.mid for i in prob.index]

    ax2 = ax.twinx()
    ax2.bar(range(len(count)), count.values, alpha=0.2, color="gray")
    ax2.set_ylabel("# hours in bin", color="gray")

    ax.plot(range(len(prob)), prob.values * 100, color="red", lw=2, marker="o", ms=4)
    ax.set_xlabel(f"{title} bins (left=drop, right=rise)")
    ax.set_ylabel("Rain probability (%)")
    ax.set_title(title)
    step = max(1, len(centers) // 8)
    ax.set_xticks(range(0, len(centers), step))
    ax.set_xticklabels([f"{centers[i]:.1f}" for i in range(0, len(centers), step)], fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "07_pressure_rain_prob.png", bbox_inches="tight")
plt.show()
"""),

M("""\
## Key Findings & Feature Engineering Plan

### What the data shows

| Signal | r with rain_next_24h | Interpretation |
|--------|----------------------|----------------|
| `pressure` | -0.080 | Low pressure = wet air mass |
| `pressure_change_6h` | -0.067 | **Falling** pressure = rain coming |
| `pressure_change_12h` | -0.062 | Stronger 12h drop = stronger signal |
| `precip_rolling_6h` | +0.049 | Current rain often continues |
| `humidity` | +0.023 | High humidity precedes rain |

### Root cause of R² = 0.002

87.3% of hours have **zero** precipitation.
A regression model minimises MSE by predicting near-zero for all rows.
The model is not wrong — it's just optimising the wrong objective for a sparse target.

### Solution: two-stage model (`train.py`)

```
Stage 1 — XGBClassifier
  Input : all features (with pressure_change_*, humidity lags, precip rolling)
  Target: has_rain_24h = (precip_24h > 0.5)
  Weight: scale_pos_weight = #no-rain / #rain   (≈ 38x)
  Output: P(rain)

Stage 2 — XGBRegressor  (trained only on rain rows)
  Input : same features, subset where precip_24h > 0.5
  Target: precip_24h
  Output: amount in mm

Combined predict:
  if P(rain) > 0.30  →  output = amount (from regressor)
  else               →  output = 0.0
```

### New features added to `features.py`

| Group | New columns |
|-------|-------------|
| Pressure changes | `pressure_change_3h/6h/12h/24h` |
| Pressure lags | `pressure_lag_1h/3h/6h` |
| Humidity lags | `humidity_lag_1h/3h/6h/12h/24h` |
| Humidity rolling | `humidity_roll_mean_6h/24h` |
| Precip rolling sum | `precip_roll_sum_3h/6h/24h` |
| Weather code | `weathercode_raw` |
"""),

]

nb["cells"] = cells
out = Path(__file__).parent.parent / "notebooks" / "eda_pattaya.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"Written: {out}")
