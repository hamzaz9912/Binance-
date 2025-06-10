import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from binance.client import Client
from datetime import datetime, timedelta
import threading
import queue
from websocket import WebSocketApp
import json

# -------------------------------
# Initialization
# -------------------------------
st.set_page_config(page_title="📈 Crypto Forecast Pro", layout="wide")


# Initialize Binance client
@st.cache_resource
def init_binance_client():
    try:
        client = Client(st.secrets["binance"]["api_key"], st.secrets["binance"]["api_secret"])
        client.ping()
        return client
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


client = init_binance_client()

# -------------------------------
# WebSocket Management
# -------------------------------
price_queue = queue.Queue()


def on_message(ws, message):
    data = json.loads(message)
    if 'p' in data:
        price = float(data['p'])
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        price_queue.put((timestamp, price))


def manage_websocket(symbol):
    if 'ws' in st.session_state and st.session_state.ws:
        st.session_state.ws.close()
    symbol = str(symbol)
    ws = WebSocketApp(f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade")
    ws.on_message = on_message
    st.session_state.ws = ws
    threading.Thread(target=ws.run_forever, daemon=True).start()


# -------------------------------
# Data Functions
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
        return pd.DataFrame([(datetime.utcfromtimestamp(k[0] / 1000), float(k[4]))
                             for k in klines], columns=['ds', 'y'])
    except:
        return pd.DataFrame()


# -------------------------------
# Forecasting Functions
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
# Main App
# -------------------------------
def main():
    st.title("⏳ Multi-Timeframe Crypto Predictor")

    # Sidebar controls
    st.sidebar.header("Configuration")
    pairs = get_usdt_pairs()
    default_index = pairs.index('BTCUSDT') if 'BTCUSDT' in pairs else 0
    selected_pair = st.sidebar.selectbox("Choose Asset", pairs, index=default_index)

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
    current_price = combined_df['y'].iloc[-1] if not combined_df.empty else None
    if current_price:
        st.metric(f"💰 {selected_pair} Current Price", f"${current_price:.2f}",
                  delta=f"{(current_price - combined_df['y'].iloc[-2]):.2f}"
                  if len(combined_df) > 1 else "N/A")

    # Forecast mode selection
    st.sidebar.subheader("Forecast Settings")
    forecast_mode = st.sidebar.radio("Forecast Type",
                                     ["Next Hour", "Custom Date"])

    if forecast_mode == "Custom Date":
        min_date = datetime.now() + timedelta(hours=1)
        max_date = datetime.now() + timedelta(days=14)
        selected_date = st.sidebar.date_input("Select Target Date",
                                              min_value=min_date,
                                              max_value=max_date)
        hours_ahead = int((selected_date - datetime.now().date()).days * 24)
        hours_ahead += (23 - datetime.now().hour)  # Adjust for current hour
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

        # Create main chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df['ds'], y=combined_df['y'],
                                 name='Historical Data', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=st.session_state.forecast['ds'],
                                 y=st.session_state.forecast['yhat'],
                                 name='Predictions', line=dict(color='#FFA15A')))
        fig.update_layout(
            title=f"{selected_pair} Price Trajectory",
            xaxis_title="Date/Time (UTC)",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create forecast breakdown
        st.subheader("📈 Forecast Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Prediction Details")
            latest_pred = st.session_state.forecast.iloc[-1]
            current_time = datetime.now().strftime("%H:%M UTC")

            if st.session_state.forecast_type == "Next Hour":
                st.metric("Next Hour Prediction",
                          f"${latest_pred['yhat']:.2f}",
                          delta=f"{(latest_pred['yhat'] - current_price):.2f} from now")
            else:
                st.metric(f"{selected_date.strftime('%Y-%m-%d')} Prediction",
                          f"${latest_pred['yhat']:.2f}",
                          delta=f"{(latest_pred['yhat'] - current_price):.2f} projected change")

        with col2:
            st.markdown("### Confidence Range")
            st.write(f"**95% Confidence Interval:**")
            st.write(f"Lower Bound: ${latest_pred['yhat_lower']:.2f}")
            st.write(f"Upper Bound: ${latest_pred['yhat_upper']:.2f}")
            st.progress(0.95, text="Prediction Confidence")

        # Detailed forecast table
        st.subheader("📅 Hour-by-Hour Forecast")
        forecast_display = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(hours_ahead)
        forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d %H:%M UTC')
        st.dataframe(
            forecast_display.style.format({
                'yhat': '{:.2f}',
                'yhat_lower': '{:.2f}',
                'yhat_upper': '{:.2f}'
            }).applymap(lambda x: 'color: #FFA15A', subset=['yhat']),
            column_config={
                'ds': 'Timestamp',
                'yhat': 'Predicted Price',
                'yhat_lower': 'Minimum Estimate',
                'yhat_upper': 'Maximum Estimate'
            },
            use_container_width=True,
            height=400
        )

    # Market overview
    st.sidebar.subheader("Market Overview")
    if st.sidebar.button("Refresh Market Data"):
        st.cache_data.clear()

    # Market overview section with error handling
    st.header("📊 Active USDT Trading Pairs")
    try:
        # Get 24-hour ticker data
        tickers_24h = client.get_ticker()
        market_df = pd.DataFrame(tickers_24h)

        # Filter and clean data
        market_df = market_df[market_df['symbol'].str.endswith('USDT')][[
            'symbol', 'lastPrice', 'highPrice', 'lowPrice',
            'priceChangePercent', 'volume', 'quoteVolume'
        ]]

        # Rename and convert columns
        market_df.columns = [
            'Pair', 'Current Price', '24h High', '24h Low',
            '24h Change (%)', 'Base Volume', 'Quote Volume'
        ]
        numeric_cols = ['Current Price', '24h High', '24h Low',
                        '24h Change (%)', 'Base Volume', 'Quote Volume']
        market_df[numeric_cols] = market_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Add safe status calculation
        def calculate_status(row):
            try:
                if row['24h High'] == 0 or row['24h Low'] == 0:
                    return 'Invalid Data'
                high_ratio = row['Current Price'] / row['24h High']
                low_ratio = row['Current Price'] / row['24h Low']
                if high_ratio > 0.98: return 'Near High'
                if low_ratio < 1.02: return 'Near Low'
                return 'Mid Range'
            except:
                return 'N/A'

        market_df['Status'] = market_df.apply(calculate_status, axis=1)

        # Filter out invalid data
        market_df = market_df[
            (market_df['24h High'] > 0) &
            (market_df['24h Low'] > 0) &
            (market_df['Status'] != 'Invalid Data')
            ]

        # Sort and format
        market_df = market_df.sort_values('Quote Volume', ascending=False)

        # Search and display
        search = st.text_input("🔍 Search pairs:")
        if search:
            market_df = market_df[market_df['Pair'].str.contains(search.upper())]

        st.dataframe(
            market_df.style.format({
                'Current Price': '{:.8f}',
                '24h High': '{:.8f}',
                '24h Low': '{:.8f}',
                '24h Change (%)': '{:.2f}%',
                'Quote Volume': '${:,.2f}'
            }).background_gradient(subset=['24h Change (%)'], cmap='RdYlGn')
            .applymap(lambda x: 'color: #2ecc71' if 'High' in str(x)
            else 'color: #e74c3c' if 'Low' in str(x)
            else '', subset=['Status'])
            .bar(subset=['Quote Volume'], color='#3498db'),
            height=600,
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")

if __name__ == "__main__":
    main()