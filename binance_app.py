import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
from binance.client import Client
from datetime import datetime, timedelta, timezone
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
        klines = client.get_historical_klines(symbol, interval, "30 days ago UTC")
        return pd.DataFrame([(datetime.utcfromtimestamp(k[0] / 1000), float(k[4]))
                              for k in klines], columns=['ds', 'y'])
    except:
        return pd.DataFrame()


# -------------------------------
# Forecasting Functions
# -------------------------------
def generate_forecast(data, periods, freq):
    forecasts = []
    scales = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different changepoint priors for varied forecasts
    for scale in scales:
        try:
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, seasonality_mode='multiplicative', changepoint_prior_scale=scale)
            model.fit(data)
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)
            forecasts.append(forecast)
        except:
            forecasts.append(pd.DataFrame())
    return forecasts


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

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

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
        periods = 10  # Forecast next 10 intervals for detailed prediction line
        freq = selected_interval["freq"]
        selected_date = datetime.now() + timedelta(minutes=selected_interval["minutes"] * periods)

    # Generate forecasts
    if not combined_df.empty and st.sidebar.button("Generate Predictions"):
        with st.spinner("Crunching numbers..."):
            forecasts = generate_forecast(combined_df, periods, freq)
            if not all(f.empty for f in forecasts):
                st.session_state.forecasts = forecasts
                st.session_state.forecast = forecasts[2]  # Use middle forecast for table
                st.session_state.combined_df = combined_df
                st.session_state.forecast_type = forecast_mode
                st.session_state.target_date = selected_date
                st.session_state.selected_interval = selected_interval_label
                st.session_state.periods = periods

    # Display results
    if 'forecast' in st.session_state:
        st.header(f"🔮 {forecast_mode} Forecast Results")

        # Prepare data
        user_tz = pytz.timezone('Asia/Karachi')
        now_tz = datetime.now(pytz.utc).astimezone(user_tz)
        combined_display = st.session_state.combined_df.copy()
        combined_display['ds'] = pd.to_datetime(combined_display['ds'])
        if combined_display['ds'].dt.tz is None:
            combined_display['ds'] = combined_display['ds'].dt.tz_localize('UTC').dt.tz_convert(user_tz)
        combined_display['ds'] = combined_display['ds'].dt.tz_convert(user_tz)
        forecast_display_df = st.session_state.forecast.tail(st.session_state.periods).copy()
        forecast_display_df['ds'] = forecast_display_df['ds'].dt.tz_localize('UTC').dt.tz_convert(user_tz)

        # Historical chart
        st.subheader("📈 Historical Price Chart")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(x=combined_display['ds'], y=combined_display['y'], name='Historical Prices', line=dict(color='blue')))
        hist_fig.update_layout(
            title=f"{selected_pair} Historical Prices",
            xaxis_title="Time (PKT)",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(hist_fig, width='stretch')

        # Forecast chart
        st.subheader("🔮 Forecast Chart")
        forecast_fig = go.Figure()
        # Show only Forecast 1
        if st.session_state.forecasts and not st.session_state.forecasts[0].empty:
            f_df = st.session_state.forecasts[0].tail(st.session_state.periods).copy()
            f_df['ds'] = f_df['ds'].dt.tz_localize('UTC').dt.tz_convert(user_tz)
            forecast_fig.add_trace(go.Scatter(x=f_df['ds'], y=f_df['yhat'], name='Forecast', line=dict(color='#FFA15A', width=2)))
        forecast_fig.update_layout(
            title=f"{selected_pair} Price Forecast",
            xaxis_title="Time (PKT)",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(forecast_fig, width='stretch')

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
        forecast_display = forecast_display_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        forecast_display.loc[:, 'ds'] = forecast_display['ds'].dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(
            forecast_display.style.format({
                'yhat': '{:.2f}',
                'yhat_lower': '{:.2f}',
                'yhat_upper': '{:.2f}'
            }).map(lambda x: 'color: #FFA15A', subset=['yhat']),
            column_config={
                'ds': 'Timestamp (PKT)',
                'yhat': 'Predicted Price',
                'yhat_lower': 'Minimum Estimate',
                'yhat_upper': 'Maximum Estimate'
            },
            width='stretch',
            height=400
        )

    # Market overview
    st.sidebar.subheader("Market Overview")
    auto_refresh_market = st.sidebar.checkbox("Auto Refresh Market Data")
    if auto_refresh_market:
        st.cache_data.clear()  # Clear cache to refresh data

    if st.sidebar.button("Refresh Market Data"):
        st.cache_data.clear()

    # Market overview section with error handling
    st.header("📊 Live USDT Trading Pairs Prices")
    if client:
        try:
            # Get live ticker data
            all_tickers = client.get_all_tickers()
            if not isinstance(all_tickers, list):
                st.error("API returned an error response. Please check your API keys and account permissions.")
                return
            market_df = pd.DataFrame(all_tickers)

            # Filter to USDT pairs
            market_df = market_df[market_df['symbol'].str.endswith('USDT')]

            # Rename columns
            market_df.columns = ['Pair', 'Current Price']

            # Convert to numeric and handle errors
            market_df['Current Price'] = pd.to_numeric(market_df['Current Price'], errors='coerce')
            market_df = market_df.dropna(subset=['Current Price'])
            market_df['Current Price'] = market_df['Current Price'].astype(float)

            # Sort by price descending and limit to top 100 for performance
            market_df = market_df.sort_values('Current Price', ascending=False).head(100)

            # Search and display
            search = st.text_input("🔍 Search pairs:")
            if search:
                market_df = market_df[market_df['Pair'].str.contains(search.upper())]

            st.dataframe(
                market_df.style.format({'Current Price': '{:.8f}'}),
                height=600
            )

        except Exception as e:
            st.error(f"Error loading live market data: {str(e)}")
    else:
        st.info("Live market data not available due to API connection issue")

if __name__ == "__main__":
    main()
