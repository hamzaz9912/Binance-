import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from datetime import datetime, timedelta, timezone
import requests

# Set this ONLY in main.py
# st.set_page_config(page_title="Forex Tracker", layout="wide")

# --- CONFIG ---
FOREX_API_KEY = st.secrets.get("forex", {}).get("api_key", "demo")
BASE_URL = "https://www.alphavantage.co/query"
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "XAUUSD"]

# --- DATA FETCHING ---
def get_daily_data(pair):
    if FOREX_API_KEY == "demo":
        st.warning("Using synthetic daily data. Add real API key in secrets.toml.")
        return generate_synthetic_data(freq="D")

    base = pair[:3]
    quote = pair[3:]

    function = "FX_DAILY" if pair not in ["XAUUSD", "XAGUSD"] else "TIME_SERIES_DAILY"
    params = {
        "function": function,
        "from_symbol": base,
        "to_symbol": quote,
        "symbol": pair,
        "apikey": FOREX_API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    key = next((k for k in data if "Time Series" in k), None)
    if not key:
        st.error("API error or rate limit exceeded.")
        return generate_synthetic_data(freq="D")

    df = pd.DataFrame(data[key]).T.reset_index()
    df.columns = ['ds', 'open', 'high', 'low', 'close', *df.columns[5:]]
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['close'])
    return df[['ds', 'y']].sort_values('ds')

def generate_synthetic_data(freq="D"):
    end = datetime.now()
    if freq == "D":
        dates = pd.date_range(end=end, periods=60, freq='D')
    else:  # Hourly
        dates = pd.date_range(end=end, periods=48, freq='H')
    values = 1.12 + np.cumsum(np.random.randn(len(dates)) * 0.002)
    return pd.DataFrame({'ds': dates, 'y': values})


# --- FORECASTING ---
def forecast_with_prophet(data, period, freq):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=period, freq=freq)
    return model.predict(future)


# --- MAIN APP ---
def main():
    st.header("📈 Forex Forecasting")

    pair = st.selectbox("Select Currency Pair", FOREX_PAIRS)
    mode = st.radio("Prediction Mode", ["Hourly (Today)", "Daily (Next N days)", "Custom Date"])

    if mode == "Hourly (Today)":
        data = generate_synthetic_data(freq="H")
        forecast = forecast_with_prophet(data, 24, 'H')
        st.subheader("⏰ Hourly Forecast (Next 24 Hours)")
    elif mode == "Daily (Next N days)":
        data = get_daily_data(pair)
        days = st.slider("Forecast Days", 1, 15, 7)
        forecast = forecast_with_prophet(data, days, 'D')
        st.subheader(f"📅 Daily Forecast ({days} days)")
    else:
        data = get_daily_data(pair)
        forecast = forecast_with_prophet(data, 30, 'D')
        custom_date = st.date_input("Select Prediction Date", value=datetime.today() + timedelta(days=3))
        selected = forecast[forecast['ds'].dt.date == custom_date]
        st.subheader("📌 Forecast for " + custom_date.strftime('%Y-%m-%d'))
        if not selected.empty:
            st.metric("Forecasted Rate", f"{selected['yhat'].values[0]:.5f}")
        else:
            st.warning("Date outside forecast range. Try a closer date.")

    # --- Display chart ---
    if not forecast.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['ds'], y=data['y'],
            name="Historical", line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            name="Forecast", line=dict(color="orange")
        ))
        st.plotly_chart(fig, use_container_width=True)
        if mode == "Hourly (Today)":
            data = generate_synthetic_data(freq="H")
            forecast = forecast_with_prophet(data, 1, 'H')  # Only 1 period, 1 hour
            st.subheader("⏰ One-Hour Forecast")

            # Get only the latest prediction (the one ahead)
            latest_pred = forecast.iloc[-1]
            st.metric("Forecasted Rate (Next Hour)", f"{latest_pred['yhat']:.5f}")

            # Optional: Show chart for context
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['ds'], y=data['y'], name="Historical", line=dict(color="blue")
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color="orange", dash='dash')
            ))
            st.plotly_chart(fig, use_container_width=True)

    if mode in ["Hourly (Today)", "Daily (Next N days)"]:
        st.dataframe(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15).style.format({
                "yhat": "{:.5f}",
                "yhat_lower": "{:.5f}",
                "yhat_upper": "{:.5f}"
            })


        )
