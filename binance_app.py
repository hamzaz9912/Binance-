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
import pytz

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
def get_historical_data(symbol, interval):
    try:
        klines = client.get_historical_klines(symbol, interval, "2 days ago UTC")
        return pd.DataFrame([(datetime.utcfromtimestamp(k[0] / 1000), float(k[4]))
                             for k in klines], columns=['ds', 'y'])
    except:
        return pd.DataFrame()


# -------------------------------
# Forecasting Functions
# -------------------------------
def generate_forecast(data, periods, freq):
    try:
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=periods, freq=freq)
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

    # Time interval selection
    interval_options = {
        "5 Minutes": {"binance": Client.KLINE_INTERVAL_5MINUTE, "freq": "5min", "minutes": 5},
        "15 Minutes": {"binance": Client.KLINE_INTERVAL_15MINUTE, "freq": "15min", "minutes": 15},
        "1 Hour": {"binance": Client.KLINE_INTERVAL_1HOUR, "freq": "H", "minutes": 60},
        "4 Hours": {"binance": Client.KLINE_INTERVAL_4HOUR, "freq": "4H", "minutes": 240},
        "1 Day": {"binance": Client.KLINE_INTERVAL_1DAY, "freq": "D", "minutes": 1440}
    }
    selected_interval_label = st.sidebar.selectbox("Time Interval", list(interval_options.keys()), index=2)  # Default to 1 Hour
    selected_interval = interval_options[selected_interval_label]

    # WebSocket management
    if 'current_pair' not in st.session_state or st.session_state.current_pair != selected_pair:
        manage_websocket(selected_pair)
        st.session_state.current_pair = selected_pair
        price_queue.queue.clear()

    # Get combined data
    hist_data = get_historical_data(selected_pair, selected_interval["binance"])
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
                                      ["Next Interval", "Custom Date"])

    if forecast_mode == "Custom Date":
        min_date = datetime.now() + timedelta(hours=1)
        max_date = datetime.now() + timedelta(days=14)
        selected_date = st.sidebar.date_input("Select Target Date",
                                              min_value=min_date,
                                              max_value=max_date)
        hours_ahead = int((selected_date - datetime.now().date()).days * 24)
        hours_ahead += (23 - datetime.now().hour)  # Adjust for current hour
        periods = int(hours_ahead * 60 / selected_interval["minutes"])
        freq = selected_interval["freq"]
    else:
        periods = max(1, selected_interval["minutes"] // 5)
        freq = '5min'
        selected_date = datetime.now() + timedelta(minutes=selected_interval["minutes"])

    # Generate forecasts
    if not combined_df.empty and st.sidebar.button("Generate Predictions"):
        with st.spinner("Crunching numbers..."):
            forecast_df = generate_forecast(combined_df, periods, freq)
            if not forecast_df.empty:
                st.session_state.forecast = forecast_df
                st.session_state.forecast_type = forecast_mode
                st.session_state.target_date = selected_date
                st.session_state.selected_interval = selected_interval_label
                st.session_state.periods = periods

    # Display results
    if 'forecast' in st.session_state:
        st.header(f"🔮 {forecast_mode} Forecast Results")

        # Convert to user timezone
        user_tz = pytz.timezone('Asia/Karachi')
        combined_df_display = combined_df.copy()
        forecast_display_df = st.session_state.forecast.copy()
        combined_df_display['ds'] = combined_df_display['ds'].dt.tz_localize('UTC').dt.tz_convert(user_tz)
        forecast_display_df['ds'] = forecast_display_df['ds'].dt.tz_localize('UTC').dt.tz_convert(user_tz)

        # Filter historical data to last 30 minutes for zoom
        now_tz = datetime.now(pytz.utc).astimezone(user_tz)
        combined_df_display = combined_df_display[combined_df_display['ds'] > now_tz - timedelta(minutes=30)]

        # Create main chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=combined_df_display['ds'], y=combined_df_display['y'],
                                 name='Historical Data', line=dict(color='#636EFA')))
        fig.add_trace(go.Scatter(x=forecast_display_df['ds'],
                                 y=forecast_display_df['yhat'],
                                 name='Predictions', line=dict(color='#FFA15A')))
        fig.update_layout(
            title=f"{selected_pair} Price Trajectory",
            xaxis_title="Date/Time (PKT)",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=500
        )
        fig.update_xaxes(tickformat='%H:%M<br>%d/%m')
        st.plotly_chart(fig, use_container_width=True)

        # Create forecast breakdown
        st.subheader("📈 Forecast Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Prediction Details")
            latest_pred = forecast_display_df.iloc[-1]
            current_time = datetime.now(pytz.utc).astimezone(user_tz).strftime("%H:%M PKT")

            if st.session_state.forecast_type == "Next Interval":
                st.metric("Next Interval Prediction",
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
        st.subheader(f"📅 Forecast Breakdown ({st.session_state.selected_interval} intervals)")
        forecast_display = forecast_display_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(st.session_state.periods)
        forecast_display['ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d %H:%M PKT')
        st.dataframe(
            forecast_display.style.format({
                'yhat': '{:.2f}',
                'yhat_lower': '{:.2f}',
                'yhat_upper': '{:.2f}'
            }).applymap(lambda x: 'color: #FFA15A', subset=['yhat']),
            column_config={
                'ds': 'Timestamp (PKT)',
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
        @st.cache_resource
        def init_binance_client():
            try:
                api_key = st.secrets["binance"]["api_key"]
                api_secret = st.secrets["binance"]["api_secret"]
                client = Client(api_key, api_secret)
                client.ping()  # This will fail if credentials are wrong
                return client
            except Exception as e:
                st.error(f"🚨 Binance API Connection Error: {e}")
                return None

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