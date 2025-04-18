import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from binance.client import Client
import numpy as np

# Load API keys from Streamlit secrets
api_key = st.secrets["binance"]["api_key"]
api_secret = st.secrets["binance"]["api_secret"]

# Initialize Binance Client
client = Client(api_key, api_secret)

# Fetch all USDT trading pairs
@st.cache_data(ttl=3600)
def get_usdt_pairs():
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']
    usdt_pairs = [s['symbol'] for s in symbols if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
    return sorted(usdt_pairs)

# Fetch historical candle data
def fetch_historical_data(symbol, interval='1h', lookback='2 days ago UTC'):
    try:
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['timestamp', 'close']]
        df['close'] = pd.to_numeric(df['close'])
        df.rename(columns={'timestamp': 'ds', 'close': 'y'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Forecast using Prophet
def forecast_crypto(df, periods=24, freq='H'):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def main():
    st.set_page_config(page_title="📈 Live Crypto Dashboard", layout="wide")
    st.title("📊 Real-Time Crypto Market Dashboard")
    st.markdown("Track **live prices**, visualize trends, and predict future values using **Prophet AI**.")

    st.sidebar.title("🔎 Options")
    coin_pairs = get_usdt_pairs()
    selected_pair = st.sidebar.selectbox("Select Crypto Pair", coin_pairs, index=coin_pairs.index('BTCUSDT'))

    forecast_mode = st.sidebar.radio("Forecast Mode", ["Live (Next 1 Hour)", "Future Date"])
    if forecast_mode == "Future Date":
        forecast_date = st.sidebar.date_input("Select Date", min_value=datetime.today(),
                                              max_value=datetime.today() + timedelta(days=7))
        hours_diff = int((forecast_date - datetime.today().date()).days * 24)
        forecast_hours = max(1, hours_diff)
    else:
        forecast_hours = 1

    # Live Price
    try:
        live_price = client.get_symbol_ticker(symbol=selected_pair)
        current_price = float(live_price['price'])
        st.metric(f"💰 {selected_pair} Current Price", f"${current_price:.2f}")
    except:
        st.error("Unable to fetch live price.")

    # Fetch historical data
    st.subheader(f"📉 {selected_pair} Market Trend (Last 48 Hours)")
    hist_df = fetch_historical_data(selected_pair)
    if hist_df.empty:
        st.stop()
    st.line_chart(hist_df.set_index('ds')['y'])

    # Forecast
    st.subheader("🔮 Forecast")
    forecast_df = forecast_crypto(hist_df, periods=forecast_hours)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_df['ds'], hist_df['y'], label="Historical", color="gray", alpha=0.7)
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label="Forecast", color="blue")
    ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'],
                    color="lightblue", alpha=0.4)
    ax.set_title(f"{selected_pair} Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Forecast Table
    st.subheader("📋 Forecast Table")
    future_rows = forecast_df[forecast_df['ds'] > hist_df['ds'].max()].head(forecast_hours)
    st.dataframe(future_rows[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'DateTime',
        'yhat': 'Predicted Price',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    }))

if __name__ == "__main__":
    main()
