import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("📊 Portfolio Performance & Risk Dashboard")

st.sidebar.header("Portfolio Inputs")

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    value="AAPL, MSFT, AMZN"
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Convert input to list
tickers_list = [t.strip() for t in tickers.split(",")]

# Download prices
raw = yf.download(
    tickers_list,
    start=start_date,
    end=end_date,
    progress=False
)

# Use Adjusted if available, else Close
if "Adj Close" in raw.columns:
    data = raw["Adj Close"]
else:
    data = raw["Close"]

st.subheader("📈 Price History")
st.line_chart(data)










st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 70px;        /* moved up so it clears the Manage toggle */
        right: 15px;
        color: #9a9a9a;
        font-size: 13px;
        font-weight: 500;
        z-index: 1000;       /* makes sure it’s always visible */
    }
    </style>

    <div class="footer">
        Built by Harbhajan — Portfolio Project
    </div>
    """,
    unsafe_allow_html=True
)
