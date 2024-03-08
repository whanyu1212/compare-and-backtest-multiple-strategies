import streamlit as st
import math
from annotated_text import annotated_text
from streamlit_lottie import st_lottie
from datetime import datetime
from util.utility_functions import (
    load_css,
    plot_sma_trend,
    plot_capital_changes,
    plot_profit_loss,
    plot_drawdown_comparison,
)
from util.st_column_config import sma_column_config
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

    for _ in range(5):
        st.text("")
    st.lottie(
        "https://lottie.host/9cfe5bca-7b5c-4957-b114-582f81e20201/2idRRfzhi2.json",
        height=100,
        width=200,
        speed=1,
        key="initial",
    )
    st.text("")
    st.text("")

    link = ":point_right: Github Repository: [link](https://github.com/whanyu1212/compare-and-backtest-multiple-strategies)"
    st.markdown(link, unsafe_allow_html=True)


extractor = FinancialDataExtractor(
    symbols=symbols,
    start=date_range[0].strftime("%Y-%m-%d"),
    end=date_range[1].strftime("%Y-%m-%d"),
    interval="1d",
)
benchmark_extractor = FinancialDataExtractor(
    symbols=["^GSPC"],
    start=date_range[0].strftime("%Y-%m-%d"),
    end=date_range[1].strftime("%Y-%m-%d"),
    interval="1d",
)
benchmark_df = benchmark_extractor.data.drop(
    ["Dividends", "Stock Splits"], axis=1
).reset_index()
benchmark_df["Price Returns"] = benchmark_df["Close"].pct_change()
left_over = (
    float(initial_capital)
    - math.floor(float(initial_capital) / benchmark_df.loc[0, "Close"])
    * benchmark_df.loc[0, "Close"]
)
benchmark_df["Total Buy and Hold Capital"] = (
    float(initial_capital) * (1 + benchmark_df["Price Returns"]).cumprod() + left_over
)
benchmark_df["Total Buy and Hold Capital"].fillna(float(initial_capital), inplace=True)


df = extractor.data.drop(["Dividends", "Stock Splits"], axis=1)

col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1, 1, 1, 1, 1])

with col1:
    st.markdown(
        f'<h2 class="my-header">Stock Backtesting Dashboard</h2>',
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
        "Triple SMA Crossover Strategy",
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
        st.dataframe(
            benchmark_df.drop("Symbol", axis=1),
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
        sma_df = sma_tester.backtesting_flow()
        sma_sharpe = sma_df["Total Strategy Capital"].pct_change().mean() / (
            sma_df["Total Strategy Capital"].pct_change().std()
        )
        sma_sortino = sma_df["Total Strategy Capital"].pct_change().mean() / (
            sma_df["Total Strategy Capital"]
            .pct_change()
            .loc[sma_df["Total Strategy Capital"].pct_change() < 0]
            .std()
        )
        sma_profit_percentage = (
            (sma_df["Total Strategy Capital"].iloc[-1] / float(initial_capital)) - 1
        ) * 100
        sma_buy_hold_percentage = (
            (sma_df["Total Buy and Hold Capital"].iloc[-1] / float(initial_capital)) - 1
        ) * 100
        sma_max_drawdown = sma_df["Strategy Max Drawdown"].iloc[-1]
        sma_total_signa = sma_df.query("Signal!='Hold'").shape[0]

        st.header("Intuition behind the Triple SMA Crossover Strategy")
        annotated_text(
            "The Triple SMA Crossover strategy involves",
            ("three Simple Moving Averages (SMAs)", "", "#fea"),
            "to signal trading opportunities based on trend direction changes. \
            A buy signal is generated when a short-term SMA ",
            ("crosses above", "", "#fea"),
            " both medium- and long-term SMAs, indicating an upward\
            trend. Conversely, a sell signal occurs when a short-term SMA",
            ("crosses below", "", "#fea"),
            " both medium- and long-term SMAs, suggesting a \
            downward trend. This method aims to filter market noise and \
            improve trade reliability by using three time frames for trend confirmation",
        )
        with st.expander("Key Statistics"):
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            with col1:
                with st.container(height=120):
                    st.markdown(
                        f"<h2 style='text-align: center; font-family: Space Grotesk;'>Sharpe Ratio: <br> {round(sma_sharpe,2)} </h2>",
                        unsafe_allow_html=True,
                    )
            with col2:
                with st.container(height=120):
                    st.markdown(
                        f"<h2 style='text-align: center; font-family: Space Grotesk;'>Sortino Ratio: <br> {round(sma_sortino,2)}</h2>",
                        unsafe_allow_html=True,
                    )

            with col3:
                with st.container(height=120):
                    st.markdown(
                        f"<h2 style='text-align: center; font-family: Space Grotesk;'>Strategy Profit %: <br> {round(sma_profit_percentage,2)}%</h2>",
                        unsafe_allow_html=True,
                    )
            with col4:
                with st.container(height=120):
                    st.markdown(
                        f"<h2 style='text-align: center; font-family: Space Grotesk;'>Buy&Hold Profit %: <br> {round(sma_buy_hold_percentage,2)}%</h2>",
                        unsafe_allow_html=True,
                    )
            with col5:
                with st.container(height=120):
                    st.markdown(
                        f"<h2 style='text-align: center; font-family: Space Grotesk;'>Max Drawdown: <br> {round(sma_max_drawdown,2)} </h2>",
                        unsafe_allow_html=True,
                    )
            with col6:
                with st.container(height=120):
                    st.markdown(
                        f"<h2 style='text-align: center; font-family: Space Grotesk;'>Total No. of Signals: <br> {sma_total_signa} </h2>",
                        unsafe_allow_html=True,
                    )
        with st.expander("Processed data"):
            st.dataframe(
                sma_df, use_container_width=True, column_config=sma_column_config
            )
        with st.expander("Key visuals"):
            col1, col2 = st.columns([1, 1])
            with col1:
                plot_sma_trend(sma_df, 3, 3)
                # plot drawdown
                plot_drawdown_comparison(sma_df)

            with col2:
                plot_capital_changes(sma_df, "skyblue", "dodgerblue")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=sma_df["Date"],
                        y=sma_df["Total Strategy Capital"],
                        name="Strategy Capital Change",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_df["Date"],
                        y=benchmark_df["Total Buy and Hold Capital"],
                        name="SP500 Buy and Hold Benchmark",
                    )
                )
                fig.update_layout(
                    title="Capital Change vs Benchmark",
                    xaxis_title="Date",
                    yaxis_title="Capital",
                )
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            plot_profit_loss(sma_df, "crimson", "lightgrey")
