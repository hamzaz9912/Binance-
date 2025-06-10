import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from binance.client import Client
from datetime import datetime, timedelta, date
import threading
import queue
from websocket import WebSocketApp
import json

# -------------------------------
# Initialization
# -------------------------------
st.set_page_config(page_title="📈 Crypto Forecast Pro", layout="wide")

# -------------------------------
# Initialize Binance Client
# -------------------------------
@st.cache_resource
def init_binance_client():
    try:
        api_key = st.secrets["binance"]["api_key"]
        api_secret = st.secrets["binance"]["api_secret"]
        client = Client(api_key, api_secret)
        client.ping()
        return client
    except Exception as e:
        st.error(f"🔐 Binance API key/secret not found or invalid. Set them in `.streamlit/secrets.toml`.\n\nError: {e}")
        return None

client = init_binance_client()

if not client:
    st.stop()

# -------------------------------
# WebSocket Management
# -------------------------------
price_queue = queue.Queue()

def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'p' in data:
            price = float(data['p'])
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            price_queue.put((timestamp, price))
    except Exception:
        pass

def manage_websocket(symbol):
    if 'ws' in st.session_state and st.session_state.ws:
        try:
            st.session_state.ws.close()
        except:
            pass
    ws = WebSocketApp(f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade", on_message=on_message)
    st.session_state.ws = ws
    threading.Thread(target=ws.run_forever, daemon=True).start()

# -------------------------------
# Data Fetch Functions
# -------------------------------
@st.cache_data(ttl=3600)
def get_usdt_pairs():
    try:
        info = client.get_exchange_info()
        return sorted([s['symbol'] for s in info['symbols']
                       if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'])
    except:
        return []

@st.cache_data(ttl=300)
def get_historical_data(symbol):
    try:
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "2 days ago UTC")
        return pd.DataFrame([(datetime.utcfromtimestamp(k[0] / 1000), float(k[4])) for k in klines], columns=['ds', 'y'])
    except:
        return pd.DataFrame()

# -------------------------------
# Forecasting Logic
# -------------------------------
def generate_forecast(data, hours_ahead):
    try:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=hours_ahead, freq='H')
        return model.predict(future)
    except:
        return pd.DataFrame()

# -------------------------------
# Streamlit App Logic
# -------------------------------
def main():
    st.title("⏳ Multi-Timeframe Crypto Predictor")

    # Sidebar controls
    st.sidebar.header("Configuration")
    pairs = get_usdt_pairs()
    if not pairs:
        st.error("Unable to fetch trading pairs. Check your Binance credentials.")
        return
    selected_pair = st.sidebar.selectbox("Choose Asset", pairs, index=pairs.index('BTCUSDT') if 'BTCUSDT' in pairs else 0)

    # WebSocket management
    if 'current_pair' not in st.session_state or st.session_state.current_pair != selected_pair:
        manage_websocket(selected_pair)
        st.session_state.current_pair = selected_pair
        price_queue.queue.clear()

    # Get combined data
    hist_data = get_historical_data(selected_pair)
    live_data = []
    while not price_queue.empty():
        live_data.append(price_queue.get())

    if live_data:
        live_df = pd.DataFrame(live_data, columns=['ds', 'y'])
        live_df['ds'] = pd.to_datetime(live_df['ds'])
        combined_df = pd.concat([hist_data, live_df]).drop_duplicates('ds').sort_values('ds')
    else:
        combined_df = hist_data

    # Real-time price display
    if not combined_df.empty:
        current_price = combined_df['y'].iloc[-1]
        prev_price = combined_df['y'].iloc[-2] if len(combined_df) > 1 else current_price
        st.metric(f"💰 {selected_pair} Current Price", f"${current_price:.2f}", delta=f"{current_price - prev_price:.2f}")
    else:
        st.warning("Not enough data to show current price.")
        return

    # Forecast Settings
    st.sidebar.subheader("Forecast Settings")
    forecast_mode = st.sidebar.radio("Forecast Type", ["Next Hour", "Custom Date"])

    if forecast_mode == "Custom Date":
        today = date.today()
        selected_date = st.sidebar.date_input("Select Target Date", min_value=today + timedelta(days=1), max_value=today + timedelta(days=14))
        hours_ahead = (selected_date - today).days * 24
        hours_ahead += (23 - datetime.now().hour)
    else:
        hours_ahead = 1
        selected_date = datetime.now() + timedelta(hours=1)

    # Generate forecasts
    if not combined_df.empty and st.sidebar.button("Generate Predictions"):
        with st.spinner("Crunching numbers..."):
            forecast_df = generate_forecast(combined_df, hours_ahead)
            if not forecast_df.empty:
                st.session_state.forecast = forecast_df
                st.session_state.forecast_type = forecast_mode
                st.session_state.target_date = selected_date

    # Display results
    if 'forecast' in st.session_state:
        st.header(f"🔮 {forecast_mode} Forecast Results")
        forecast_df = st.session_state.forecast
        latest_pred = forecast_df.iloc[-1]

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df['ds'], y=combined_df['y'], name='Historical', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(color='orange')))
        fig.update_layout(title=f"{selected_pair} Price Forecast", xaxis_title="Time", yaxis_title="Price", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Prediction Summary
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Prediction Summary")
            label = "Next Hour Prediction" if forecast_mode == "Next Hour" else f"{selected_date.strftime('%Y-%m-%d')} Prediction"
            st.metric(label, f"${latest_pred['yhat']:.2f}", delta=f"{latest_pred['yhat'] - current_price:.2f}")

        with col2:
            st.markdown("### Confidence Interval")
            st.write(f"**95% CI**: ${latest_pred['yhat_lower']:.2f} → ${latest_pred['yhat_upper']:.2f}")
            st.progress(0.95, text="Prediction Confidence")

        # Forecast Table
        st.subheader("📅 Forecast Details")
        forecast_display = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(hours_ahead)
        forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d %H:%M UTC')
        st.dataframe(
            forecast_display.style.format({
                'yhat': '{:.2f}',
                'yhat_lower': '{:.2f}',
                'yhat_upper': '{:.2f}'
            }),
            use_container_width=True,
            height=400
        )

    # Market Overview
    st.sidebar.subheader("Market Overview")
    if st.sidebar.button("Refresh Market Data"):
        st.cache_data.clear()

    st.header("📊 Top USDT Trading Pairs")
    try:
        tickers_24h = client.get_ticker()
        market_df = pd.DataFrame(tickers_24h)
        market_df = market_df[market_df['symbol'].str.endswith('USDT')]

        market_df = market_df[['symbol', 'lastPrice', 'highPrice', 'lowPrice', 'priceChangePercent', 'volume', 'quoteVolume']]
        market_df.columns = ['Pair', 'Current Price', '24h High', '24h Low', '24h Change (%)', 'Base Volume', 'Quote Volume']
        for col in market_df.columns[1:]:
            market_df[col] = pd.to_numeric(market_df[col], errors='coerce')

        def status(row):
            if row['24h High'] == 0 or row['24h Low'] == 0:
                return 'Invalid'
            high_ratio = row['Current Price'] / row['24h High']
            low_ratio = row['Current Price'] / row['24h Low']
            if high_ratio > 0.98: return 'Near High'
            elif low_ratio < 1.02: return 'Near Low'
            return 'Stable'

        market_df['Status'] = market_df.apply(status, axis=1)
        market_df = market_df.dropna().sort_values('Quote Volume', ascending=False)

        search = st.text_input("🔍 Search Pairs:")
        if search:
            market_df = market_df[market_df['Pair'].str.contains(search.upper())]

        st.dataframe(
            market_df.style.format({
                'Current Price': '{:.6f}',
                '24h High': '{:.6f}',
                '24h Low': '{:.6f}',
                '24h Change (%)': '{:.2f}%',
                'Quote Volume': '${:,.0f}'
            }),
            use_container_width=True,
            height=600
        )
    except Exception as e:
        st.error(f"Market data error: {e}")

if __name__ == "__main__":
    main()
