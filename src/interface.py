import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
from fetch_data import FinancialDataExtractor
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go


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

tab1, tab2, tab3 = st.tabs(
    ["Exploratory Data Analysis", "Strategy Comparison", "Strategy Optimization"]
)


def fetch_data():
    extractor = FinancialDataExtractor(
        symbol=selected_stock,
        start=date_range[0].strftime("%Y-%m-%d"),
        end=date_range[1].strftime("%Y-%m-%d"),
        amount=10000,
        transaction_cost=0.01,
        interval="1d",
    )

    df, df_stats = extractor.data_extraction_flow()

    return df, df_stats


with sidebar:
    st.subheader("Backtesting Web App")
    st.divider()
    selected_stock = st.sidebar.text_input("Enter a valid stock ticker...", "AAPL")
    st.text("")
    date_range = st.date_input(
        "Select date range for data extraction",
        [datetime(2015, 1, 1), datetime(2019, 12, 31)],
    )
    st.text("")
    button_clicked = st.sidebar.button("GO")


with tab1:

    if button_clicked:
        with st.spinner("Fetching data in progress..."):
            df, df_stats = fetch_data()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("At Close", f"${round(df['Close'][-1], 2)}")
            col2.metric("Beta", f"{round(df_stats['beta'], 2)}")
            col3.metric("Volumne", f"{df_stats['volume']}")
            col4.metric("Forward PE", f"{round(df_stats['forwardPE'], 2)}")
            style_metric_cards(border_left_color="#ADD8E6")
            col_1, col_2 = st.columns(2)
            with col_1:
                st.dataframe(df, width=800, height=400)
            with col_2:
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                        )
                    ]
                )
                fig.update_layout(
                    title="Candlestick Chart of AAPL Stock's Price",
                    title_x=0.3,
                    title_y=1,
                    xaxis_title="Date",
                    yaxis_title="Closing Price",
                    xaxis_showgrid=True,  # Add grid to x-axis
                    yaxis_showgrid=True,  # Add grid to y-axis
                    plot_bgcolor="rgba(0, 0, 0, 0)",  # Set plot background to transparent
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
