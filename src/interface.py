import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
from fetch_data import FinancialDataExtractor
# Global Variables
st.set_page_config(layout="wide")
st_lottie(
    "https://lottie.host/added42b-a2a9-42da-a297-744f34ec6533/ovGKQ7QQKu.json",
    height=400,
    width=800,
    speed=1,
    # key="initial",
)
st.title("Backtesting different trading strategies")

sidebar = st.sidebar
st.text("")
st.text("")

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Strategy Comparison", "Strategy Optimization"])
def fetch_data():
    df = FinancialDataExtractor(
        symbol=selected_stock,
        start=date_range[0].strftime("%Y-%m-%d"),
        end=date_range[1].strftime("%Y-%m-%d"),
        amount=10000,
        transaction_cost=0.01,
        interval="1d",
    ).data

    return df

with sidebar:
    st.subheader("Backtesting Web App")
    st.divider()
    selected_stock = st.sidebar.text_input("Enter a valid stock ticker...", "AAPL")
    st.text("")
    date_range = st.date_input("Select date range for data extraction", [datetime(2015, 1, 1), datetime(2019, 12, 31)])
    st.text("")
    button_clicked = st.sidebar.button("GO")



with tab1:
    if button_clicked:
        with st.spinner('Fetching data in progress...'):
            df = fetch_data()
            st.dataframe(df)

