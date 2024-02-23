import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
from util.utility_functions import fetch_data, plot_candlestick_with_indicators
from calculate_indicators import FinancialIndicators
from fetch_data import FinancialDataExtractor
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
from lightweight_charts.widgets import StreamlitChart

st.set_page_config(layout="wide")

symbols = ["AAPL", "GOOGL", "META", "NFLX", "AMZN"]


def custom_header(text):
    st.markdown(
        f'<h1 style="font-family:Space Grotesk; font-size:35px; font-weight:700; line-height:1.2; text-align:left; padding-bottom:35px;">{text}</h1>',
        unsafe_allow_html=True,
    )


sidebar = st.sidebar
with sidebar:
    st.subheader("Chart Settings")
    st.divider()
    option = st.selectbox(
        "Pick the stock to update the chart",
        symbols,
    )
    st.text("")
    date_range = st.date_input(
        "Backtest timeframe",
        [datetime(2015, 1, 1), datetime(2019, 12, 31)],
    )
    st.text("")
    interval = st.selectbox("Select the interval", ["1d", "1wk", "1mo"])
    button_clicked = st.sidebar.button("Update Chart")


extractor = FinancialDataExtractor(
    symbols=symbols,
    start=date_range[0].strftime("%Y-%m-%d"),
    end=date_range[1].strftime("%Y-%m-%d"),
    interval="1d",
)
df = extractor.data.drop(["Dividends", "Stock Splits"], axis=1)

col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1, 1, 1, 1])

with col1:
    custom_header("Stock Trading Dashboard")


columns = [col2, col3, col4, col5, col6]

for symbol, column in zip(symbols, columns):
    filtered_df = df[df["Symbol"] == symbol].reset_index(drop=True)
    percentage_change = (
        filtered_df["Close"].iloc[-1] - filtered_df["Close"].iloc[-2]
    ) / filtered_df["Close"].iloc[-2]
    percentage_string = "{:.2f}%".format(percentage_change * 100)
    column.metric(
        symbol,
        round(filtered_df["Close"].iloc[-1], 2),
        percentage_string,
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Exploratory Data Analysis",
        "SuperTrend Strategy",
        "Disparity Index Strategy",
        "True Strength Index Strategy",
        "ML based Strategy",
    ]
)

style_metric_cards(border_left_color="#ADD8E6")
with tab1:
    if button_clicked:
        with st.container():
            chart = StreamlitChart(height=500)
            chart.grid(vert_enabled=True, horz_enabled=True)

            chart.layout(
                background_color="#131722", font_family="Trebuchet MS", font_size=16
            )

            chart.candle_style(
                up_color="#2962ff",
                down_color="#e91e63",
                border_up_color="#2962ffcb",
                border_down_color="#e91e63cb",
                wick_up_color="#2962ffcb",
                wick_down_color="#e91e63cb",
            )

            chart.volume_config(up_color="#2962ffcb", down_color="#e91e63cb")
            chart.legend(
                visible=True, font_family="Trebuchet MS", ohlc=True, percent=True
            )

            chart.set(df.query(f"Symbol == '{option}'"))
            chart.load()
        st.dataframe(df.query(f"Symbol == '{option}'"), use_container_width=True)
    else:
        st.info(
            "Please click on the Update Chart button to load the candlestick chart for the selected stock",
            icon="ℹ️",
        )
