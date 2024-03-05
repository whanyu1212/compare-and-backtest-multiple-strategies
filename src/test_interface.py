import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
from util.utility_functions import (
    load_css,
)
from backtests.SMA import SMAVectorBacktester
from calculate_indicators import FinancialIndicators
from fetch_data import FinancialDataExtractor
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
from lightweight_charts.widgets import StreamlitChart

st.set_page_config(layout="wide")

symbols = ["AAPL", "GOOGL", "META", "NFLX", "AMZN"]


load_css("./src/style.css")


sidebar = st.sidebar
with sidebar:
    st.subheader("Chart Settings")
    st.divider()
    option = st.selectbox(
        "Pick the stock to update the chart",
        symbols,
    )
    st.text("")
    initial_capital = st.text_input(
        "Starting Capital (Please enter a valid number)", 10000
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
    st.markdown(
        f'<h1 class="my-header">Stock Backtesting Dashboard</h1>',
        unsafe_allow_html=True,
    )


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
        "SMA 5-8-13 Strategy",
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
            st.divider()

        table_title_text = f"Sample data for {option} Stock from {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
        st.markdown(
            f'<h2 class="my-header">{table_title_text}</h2>', unsafe_allow_html=True
        )
        st.dataframe(
            df.query(f"Symbol == '{option}'").drop("Symbol", axis=1),
            use_container_width=True,
        )
    else:
        st.info(
            "Please click on the Update Chart button to load the candlestick chart for the selected stock",
            icon="ℹ️",
        )

with tab2:
    if button_clicked:
        sma_tester = SMAVectorBacktester(
            df.query("Symbol==@option"), float(initial_capital)
        )
        tested_df = sma_tester.backtesting_strategy()
        # st.dataframe(tested_df)
        with st.expander("Key visuals"):
            col1, col2 = st.columns([1, 1])
            with col1:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=tested_df["Date"],
                        y=tested_df["Cumulative Strategy Return"],
                        mode="lines",
                        name="Strategy Return",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=tested_df["Date"],
                        y=tested_df["Cumulative Buy and Hold Return"],
                        mode="lines",
                        name="Buy and Hold Return",
                    )
                )
                fig.update_layout(
                    title="Strategy vs Buy and Hold Return",
                    xaxis_title="Backtesting Period",
                    yaxis_title="Cumulative Return in %",
                    title_x=0.4,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                figure = go.Figure()
                figure.add_trace(
                    go.Scatter(
                        x=tested_df["Date"],
                        y=tested_df["Strategy Drawdown"],
                        mode="lines",
                        name="Strategy Drawdown",
                    )
                )
                figure.add_trace(
                    go.Scatter(
                        x=tested_df["Date"],
                        y=tested_df["Buy and Hold Drawdown"],
                        mode="lines",
                        name="Buy and Hold Drawdown",
                    )
                )
                figure.update_layout(
                    title="Strategy vs Buy and Hold Drawdown",
                    xaxis_title="Backtesting Period",
                    yaxis_title="Drawdown in %",
                    title_x=0.4,
                )
                st.plotly_chart(figure, use_container_width=True)

        with st.expander("Statistics"):
            st.write("Values Assumed: Benchmark = S&P 500, Risk-Free Rate = 0.01")
            container = st.container(border=True)
            with container:
                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col2:
                    st.markdown("### Sharpe Ratio")
                    st.markdown(
                        tested_df["Strategy Equity Return"].mean()
                        / tested_df["Strategy Equity Return"].std(),
                    )
                    st.metric(
                        "Sharpe Ratio",
                        tested_df["Strategy Equity Return"].mean()
                        / tested_df["Strategy Equity Return"].std(),
                    )
