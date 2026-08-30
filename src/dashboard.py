"""Streamlit dashboard: historical temperature + live 24h predictions."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = Path("models")
PREDICTIONS_PATH = Path("data/processed/latest_predictions.csv")
DB_PATH = Path("data/weather.db") if not (p := Path(".env")).exists() else None


def _load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_actuals(days: int = 7) -> pd.DataFrame:
    """Load last N days of actual temperature from SQLite."""
    import os
    db_path = os.getenv("DB_PATH", "data/weather.db")
    if not Path(db_path).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"SELECT time, temp FROM weather ORDER BY time DESC LIMIT {days * 24}",
        conn,
    )
    conn.close()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time")


def _load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"])
    return df


def _run_pipeline() -> str:
    """Run fetch_live → predict via subprocess and return combined output."""
    output_lines = []
    for script in ["src/fetch_live.py", "src/predict.py"]:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
        )
        output_lines.append(f"=== {script} ===")
        output_lines.append(result.stdout or "(no output)")
        if result.stderr:
            output_lines.append(f"STDERR: {result.stderr}")
    return "\n".join(output_lines)


def _production_badge(metadata: dict) -> None:
    if metadata.get("production_ready"):
        st.success("Production ready ✓  (temp MAE < 2.0°C)")
    else:
        st.warning("Not production ready yet  (temp MAE ≥ 2.0°C)")


def main() -> None:
    st.set_page_config(page_title="Pattaya Weather Forecast", layout="wide")
    st.title("🌤 Pattaya — Weather Forecast ML")

    metadata = _load_metadata()

    # --- sidebar ---
    with st.sidebar:
        st.header("Model info")
        if metadata:
            st.metric("City", metadata.get("city", "—"))
            st.metric("Temp MAE", f"{metadata.get('mae_temp', '—'):.3f} °C")
            st.metric("Precip MAE", f"{metadata.get('mae_precip', '—'):.3f} mm")
            st.metric("Wind MAE", f"{metadata.get('mae_wind', '—'):.3f} m/s")
            trained_at = metadata.get("trained_at", "—")[:19].replace("T", " ")
            st.metric("Last trained", trained_at)
            st.metric("Train rows", f"{metadata.get('train_rows', 0):,}")
            _production_badge(metadata)
        else:
            st.info("No metadata found. Run train.py first.")

        st.divider()
        if st.button("🔄 Refresh (fetch live + predict)"):
            with st.spinner("Running pipeline..."):
                out = _run_pipeline()
            st.code(out)

    # --- main area ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Last 7 days — actual vs predicted temperature")
        actuals = _load_actuals(days=7)
        preds = _load_predictions()

        fig = go.Figure()

        if not actuals.empty:
            fig.add_trace(go.Scatter(
                x=actuals["time"],
                y=actuals["temp"],
                name="Actual temperature",
                line={"color": "#4C9BE8", "width": 2},
            ))

        if not preds.empty:
            fig.add_trace(go.Scatter(
                x=preds["timestamp"],
                y=preds["pred_temp"],
                name="Predicted temperature",
                line={"color": "#F4A261", "width": 2, "dash": "dash"},
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([preds["timestamp"], preds["timestamp"].iloc[::-1]]),
                y=pd.concat([preds["temp_upper"], preds["temp_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(244,162,97,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                name="Confidence band",
                showlegend=True,
            ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            legend={"orientation": "h"},
            margin={"t": 20},
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("24h forecast table")
        if not preds.empty:
            display = preds[["timestamp", "pred_temp", "pred_precip", "pred_windspeed"]].copy()
            display.columns = ["Time", "Temp °C", "Precip mm", "Wind m/s"]
            display["Time"] = display["Time"].dt.strftime("%m-%d %H:%M")
            st.dataframe(display, hide_index=True, height=360)
        else:
            st.info("No predictions found. Run predict.py first.")

    # --- precipitation & wind charts ---
    if not preds.empty:
        st.subheader("24h precipitation & wind forecast")
        c1, c2 = st.columns(2)

        with c1:
            fig_p = go.Figure(go.Bar(
                x=preds["timestamp"],
                y=preds["pred_precip"],
                name="Precipitation (mm)",
                marker_color="#74B3CE",
            ))
            fig_p.update_layout(yaxis_title="mm", height=250, margin={"t": 10})
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            fig_w = go.Figure(go.Scatter(
                x=preds["timestamp"],
                y=preds["pred_windspeed"],
                fill="tozeroy",
                name="Wind (m/s)",
                line={"color": "#A8DADC"},
            ))
            fig_w.update_layout(yaxis_title="m/s", height=250, margin={"t": 10})
            st.plotly_chart(fig_w, use_container_width=True)


if __name__ == "__main__":
    main()
