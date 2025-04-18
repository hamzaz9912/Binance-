import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from binance.client import Client
from datetime import datetime, timedelta
import time
import threading
import queue
from websocket import WebSocketApp
import json

st.set_page_config(layout="wide")

# -------------------------------
# API Setup
# -------------------------------
api_key = "VKuSdnSh1LK2apOk0dSaRHPvgoNcv2DpzL205A1MgA2KksTRV6FtfbPKQa4Vzq4z"
api_secret = "E6xR3mJDdknnmL2VDftz1sBpDFIUZ1EaREr6Nf14PYQ3vcg2fMNoIUNQrGZ03joo"
client = Client(api_key, api_secret)

# -------------------------------
# WebSocket Setup for Live Data
# -------------------------------
price_queue = queue.Queue()


def on_message(ws, message):
    data = json.loads(message)
    if 'p' in data:
        price = data['p']
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        price_queue.put((timestamp, float(price)))


def websocket_thread(symbol):
    ws = WebSocketApp(f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade",
                      on_message=on_message)
    st.session_state.ws = ws  # Store WebSocket instance in session state
    ws.run_forever()


# -------------------------------
# Initialize Session State
# -------------------------------
if 'ws' not in st.session_state:
    st.session_state.ws = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None


# -------------------------------
# Get All USDT Trading Pairs
# -------------------------------
@st.cache_data(ttl=3600)
def get_usdt_pairs():
    exchange_info = client.get_exchange_info()
    return [symbol['symbol'] for symbol in exchange_info['symbols']
            if symbol['symbol'].endswith('USDT') and symbol['status'] == 'TRADING']


# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("📊 Crypto Dashboard")
usdt_pairs = get_usdt_pairs()
selected_symbol = st.sidebar.selectbox("🪙 Select a Coin", usdt_pairs)
view_option = st.sidebar.radio("📌 Select View", ["Real-Time Prediction", "Future Forecast"])

# -------------------------------
# Manage WebSocket Connection
# -------------------------------
if st.session_state.current_symbol != selected_symbol:
    # Close existing connection if it exists
    if st.session_state.ws:
        st.session_state.ws.close()
        st.session_state.ws = None

    # Start new WebSocket connection
    st.session_state.current_symbol = selected_symbol
    threading.Thread(
        target=websocket_thread,
        args=(selected_symbol,),
        daemon=True
    ).start()


# -------------------------------
# Data Processing
# -------------------------------
@st.cache_data
def get_historical_data(symbol):
    try:
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "2 days ago UTC")
        return [[datetime.utcfromtimestamp(k[0] / 1000), float(k[4])] for k in klines]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return []


historical_data = get_historical_data(selected_symbol)
live_data = []
while not price_queue.empty():
    live_data.append(price_queue.get())

# Combine and process data
combined_data = historical_data + [[datetime.strptime(d[0], "%Y-%m-%d %H:%M:%S"), d[1]] for d in live_data]
df = pd.DataFrame(combined_data, columns=["ds", "y"]).drop_duplicates("ds").sort_values("ds")


# -------------------------------
# Prediction Views
# -------------------------------
def create_prediction_chart(df, periods, title):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='H')
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prediction'))
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name='Actual Price', line=dict(color='orange')))
    fig.update_layout(title=title,
                      xaxis_title="Time (UTC)",
                      yaxis_title="Price (USD)",
                      showlegend=True)
    return fig


if view_option == "Real-Time Prediction":
    st.subheader(f"📈 Real-Time {selected_symbol} Price Prediction")
    fig = create_prediction_chart(df, 12, f"Live {selected_symbol} Price Prediction")
    st.plotly_chart(fig, use_container_width=True)

elif view_option == "Future Forecast":
    st.subheader(f"🔮 {selected_symbol} Price Forecast")
    future_date = st.sidebar.date_input("📅 Select Future Date", min_value=datetime.today().date() + timedelta(days=1))
    hours_ahead = int(((future_date - datetime.today().date()).days * 24))
    fig = create_prediction_chart(df, hours_ahead, f"{selected_symbol} Price Forecast until {future_date}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Market Data Table
# -------------------------------
st.subheader("📊 All USDT Trading Pairs on Binance")


@st.cache_data(ttl=300)
def get_market_data():
    tickers = client.get_ticker_24hr()
    df = pd.DataFrame(tickers)
    usdt_df = df[df['symbol'].isin(usdt_pairs)][['symbol', 'lastPrice', 'priceChangePercent', 'volume']]
    return usdt_df.sort_values('volume', ascending=False)


market_data = get_market_data().rename(columns={
    'symbol': 'Pair',
    'lastPrice': 'Price',
    'priceChangePercent': '24h Change',
    'volume': 'Volume'
})

# Convert numeric values
market_data[['Price', '24h Change', 'Volume']] = market_data[['Price', '24h Change', 'Volume']].apply(pd.to_numeric,
                                                                                                      errors='coerce')

# Add search and filtering
search_query = st.text_input("🔍 Search Pair:")
if search_query:
    market_data = market_data[market_data['Pair'].str.contains(search_query.upper())]

# Display formatted table
st.dataframe(
    market_data.style.format({
        'Price': "{:.8f}",
        '24h Change': "{:.2f}%",
        'Volume': "${:,.2f}"
    }).background_gradient(subset=['24h Change'], cmap='RdYlGn')
    .bar(subset=['Volume'], color='#5fba7d'),
    height=800,
    use_container_width=True
)