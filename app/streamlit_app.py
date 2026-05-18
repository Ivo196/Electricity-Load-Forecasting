"""Streamlit dashboard for forecast metrics and artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.csv"
FORECAST_FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "lstm_test_forecast.png"


st.set_page_config(
    page_title="Electricity Load Forecasting",
    layout="wide",
)

st.title("Electricity Load Forecasting")

metrics_col, context_col = st.columns([2, 1])

with metrics_col:
    st.subheader("Final Test Metrics")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        display_metrics = metrics.copy()
        for column in ["mae", "rmse", "mape"]:
            if column in display_metrics.columns:
                display_metrics[column] = display_metrics[column].map(lambda value: f"{value:.2f}")
        st.table(display_metrics)
    else:
        st.info("Run `python -m src.evaluate` to generate reports/metrics.csv.")

with context_col:
    st.subheader("Project Setup")
    st.write("Target: `nat_demand`")
    st.write("Frequency: hourly")
    st.write("Split: train / validation / test")
    st.write("Lookback: 168 hours")

st.subheader("LSTM Forecast Plot")
if FORECAST_FIGURE_PATH.exists():
    st.image(str(FORECAST_FIGURE_PATH), use_container_width=True)
else:
    st.info("Train the LSTM and run evaluation to generate the forecast plot.")

st.subheader("Implemented Models")
st.write(
    "Naive previous day, seasonal naive previous week, linear regression, "
    "random forest, optional XGBoost, and LSTM."
)
