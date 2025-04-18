import yfinance as yf
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Function to fetch live data for a given symbol
def get_live_data(symbol):
    ticker = yf.Ticker(symbol)
    try:
        # Fetch live data (most recent data)
        data = ticker.history(period="1d")  # Fetches the latest day's data
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Function to get historical data for forecasting using Prophet
def get_historical_data(symbol, start="2020-01-01", end="2023-01-01"):
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        return None
    data = data.reset_index()  # Reset index to make 'Date' a column
    return data[['Date', 'Close']]  # We only care about Date and Close price

# Function to forecast using Prophet
def forecast_crypto(data, forecast_days=30):
    # Prepare the data for Prophet
    df = data.rename(columns={'Date': 'ds', 'Close': 'y'})

    # Initialize the Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    # Make a future dataframe for prediction
    future = model.make_future_dataframe(df, periods=forecast_days)

    # Get the forecast
    forecast = model.predict(future)

    return forecast

# Streamlit interface
def main():
    st.title("Crypto Trading Forecasting App")

    # List of cryptocurrency symbols (can be expanded)
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']

    # Let the user select a cryptocurrency
    selected_coin = st.selectbox("Select a cryptocurrency", crypto_symbols)

    # Fetch live data for the selected coin
    st.write(f"Fetching live data for {selected_coin}...")
    live_data = get_live_data(selected_coin)

    if live_data is not None:
        st.write(f"Live data for {selected_coin}:")
        st.write(live_data)  # Show the live data in a table

        # Show closing price as an example
        st.write(f"Closing Price: {live_data['Close'].iloc[-1]}")

        # Show a plot of the Close prices over the last 5 days
        st.write("Last 5 Days Close Prices:")
        fig = live_data['Close'].plot(kind='line', title=f"{selected_coin} - Last 5 Days", figsize=(10, 6))
        st.pyplot(fig)
    else:
        st.write(f"No data available for {selected_coin}. Please try a different symbol.")

    # Fetch historical data for forecasting
    st.write(f"Fetching historical data for forecasting...")
    historical_data = get_historical_data(selected_coin)

    if historical_data is not None and len(historical_data) > 1:
        # Forecast data using Prophet
        st.write("Making 30-day forecast...")
        forecast = forecast_crypto(historical_data)

        # Plot the forecast
        st.write("Forecast for the next 30 days:")

        # Plot the forecast
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast['ds'], forecast['yhat'], label="Forecast")
        ax.plot(forecast['ds'], forecast['yhat_lower'], linestyle='--', label="Lower Bound")
        ax.plot(forecast['ds'], forecast['yhat_upper'], linestyle='--', label="Upper Bound")

        # Highlight the historical data
        ax.plot(historical_data['Date'], historical_data['Close'], color='black', label="Historical Data", alpha=0.5)

        # Customize the plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f"Forecast for {selected_coin} - Next 30 Days")
        ax.legend()

        # Show plot
        st.pyplot(fig)

        # Show the forecast table
        st.write("Forecast Table:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

    else:
        st.write("Not enough historical data to generate the forecast. Please try a different symbol or time range.")

# Run the app
if __name__ == "__main__":
    main()
