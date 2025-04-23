import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from datetime import datetime, timedelta, timezone
import requests
import numpy as np

# Forex configuration
FOREX_API_KEY = st.secrets.get("forex", {}).get("api_key", "demo")
BASE_URL = "https://www.alphavantage.co/query"

# Added XAUUSD (Gold) as a new brand
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "XAUUSD"]


def get_forex_data(pair):
    """Fetch daily data with proper function based on asset type."""
    try:
        if FOREX_API_KEY == "demo":
            st.warning("Using synthetic data - Add real API key in secrets.toml")
            return generate_synthetic_data()

        base = pair[:3]
        quote = pair[3:]

        # Determine function
        if pair in ["XAUUSD", "XAGUSD"]:
            function = "TIME_SERIES_DAILY"
            params = {
                "function": function,
                "symbol": pair,
                "apikey": FOREX_API_KEY,
                "outputsize": "compact",
                "datatype": "json"
            }
        else:
            function = "FX_DAILY"
            params = {
                "function": function,
                "from_symbol": base,
                "to_symbol": quote,
                "apikey": FOREX_API_KEY,
                "outputsize": "compact",
                "datatype": "json"
            }

        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise Exception(f"API Error: {data['Error Message']}")
        if "Note" in data:
            raise Exception(f"Rate Limit: {data['Note']}")

        # Parse time series
        time_series_key = next((k for k in data if "Time Series" in k), None)
        if not time_series_key:
            raise Exception("No time series data found in response.")

        df = pd.DataFrame(data[time_series_key]).T.reset_index()
        df.columns = ['ds', 'open', 'high', 'low', 'close', *df.columns[5:]]
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['close'])
        return df[['ds', 'y']].sort_values('ds')

    except Exception as e:
        st.error(f"Data fetch failed: {str(e)}")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate realistic daily synthetic data"""
    end_date = datetime.now(timezone.utc)
    dates = pd.date_range(end=end_date, periods=30, freq='D')
    prices = 1.12 + np.cumsum(np.random.randn(30) * 0.005)
    return pd.DataFrame({'ds': dates, 'y': prices})


def generate_forex_forecast(data, periods=7):
    """Generate forecast using Prophet"""
    try:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=periods, freq='D')  # Daily forecast
        return model.predict(future)
    except Exception as e:
        st.error(f"Forecast error: {str(e)}")
        return pd.DataFrame()


def main():
    st.title("📈 Forex Market Tracker")

    # Initialize session state
    if 'forex_data' not in st.session_state:
        st.session_state.forex_data = pd.DataFrame()

    # Sidebar controls
    selected_pair = st.sidebar.selectbox("Select Forex/Commodity Pair", FOREX_PAIRS)
    forecast_days = st.sidebar.slider("Forecast Days", 1, 14, 7)

    # Load data
    with st.spinner("Loading forex data..."):
        st.session_state.forex_data = get_forex_data(selected_pair)

    # Display current rate
    if not st.session_state.forex_data.empty:
        current_rate = st.session_state.forex_data['y'].iloc[-1]
        st.metric(f"Current {selected_pair[:3]}/{selected_pair[3:]} Rate", f"{current_rate:.5f}")

    # Price chart
    st.subheader(f"{selected_pair[:3]}/{selected_pair[3:]} Price Movement")
    if not st.session_state.forex_data.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.forex_data['ds'],
            y=st.session_state.forex_data['y'],
            name='Actual Rate',
            line=dict(color='#1f77b4', width=1.5)
        ))
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Forecast section
    st.sidebar.subheader("Forecast Settings")
    if st.sidebar.button("Generate Forecast"):
        with st.spinner("Creating forecast..."):
            forecast = generate_forex_forecast(st.session_state.forex_data, forecast_days)

            if not forecast.empty:
                st.subheader("🔮 Exchange Rate Forecast")

                # Forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name='Forecast',
                    line=dict(color='#ff7f0e')
                ))
                fig.add_trace(go.Scatter(
                    x=st.session_state.forex_data['ds'],
                    y=st.session_state.forex_data['y'],
                    name='Historical',
                    line=dict(color='#1f77b4')
                ))
                fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Rate")
                st.plotly_chart(fig, use_container_width=True)

                # Forecast table
                st.subheader("📋 Forecast Details")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
                forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    forecast_display.style.format({
                        'yhat': '{:.5f}',
                        'yhat_lower': '{:.5f}',
                        'yhat_upper': '{:.5f}'
                    }),
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
