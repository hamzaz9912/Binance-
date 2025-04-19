import streamlit as st
import binance_app
import forex_app  # Make sure this file exists in the same directory


def main():
    st.sidebar.title("Market Selector")
    app_choice = st.sidebar.radio("Choose Platform", ["Binance", "Forex"])

    if app_choice == "Binance":
        binance_app.main()
    elif app_choice == "Forex":
        forex_app.main()  # Calls the main() from forex_app.py


if __name__ == "__main__":
    main()