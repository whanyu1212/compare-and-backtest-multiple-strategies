import streamlit as st
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from streamlit_lottie import st_lottie
from loguru import logger
from colorama import Fore, Style
from annotated_text import annotated_text
from datetime import datetime
from util.utility_functions import (
    load_css,
    create_container,
    plot_sma_trend,
    plot_dc_trend,
    plot_supertrend,
    plot_capital_changes,
    plot_drawdown_comparison,
    plot_shap_bar_chart,
    plot_returns_vs_volatility,
    plot_portfolio_weights,
)
from util.st_column_config import sma_column_config
from backtests.SuperTrend import SuperTrendVectorBacktester
from backtests.SMA import SMAVectorBacktester
from backtests.DC import DonchianChannelVectorBacktester
from backtests.ML import MLClassifierVectorBacktester
from backtests.LSTM import LSTMVectorBacktester
from portfolio_stats import PortfolioStatistics
from fetch_data import FinancialDataExtractor
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go
from lightweight_charts.widgets import StreamlitChart

st.set_page_config(layout="wide")

symbols = ["META", "AAPL", "AMZN", "NFLX", "GOOGL"]


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
        "Starting Capital (Please enter a valid number)", 100000
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
        height=150,
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
    symbols=["SPY"],
    start=date_range[0].strftime("%Y-%m-%d"),
    end=date_range[1].strftime("%Y-%m-%d"),
    interval="1d",
)
benchmark_df = benchmark_extractor.data.drop(
    ["Dividends", "Stock Splits"], axis=1
).reset_index()
benchmark_df["Market Returns"] = benchmark_df["Close"].pct_change()
left_over = (
    float(initial_capital)
    - math.floor(float(initial_capital) / benchmark_df.loc[0, "Close"])
    * benchmark_df.loc[0, "Close"]
)
benchmark_df["Buy and Hold Portfolio Value"] = (
    float(initial_capital) * (1 + benchmark_df["Market Returns"]).cumprod() + left_over
)
benchmark_df["Buy and Hold Portfolio Value"].fillna(
    float(initial_capital), inplace=True
)


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
    percentage_string = "{:.2f}%".format(percentage_change * 150)
    column.metric(
        symbol,
        round(filtered_df["Close"].iloc[-1], 2),
        percentage_string,
    )

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Pricing Visuals",
        "Portfolio Optimization",
        "SuperTrend Strategy",
        "Triple SMA Crossover Strategy",
        "Donchian Channel Strategy",
        "ML based Strategy",
        "LSTM based Strategy",
        "Overview",
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
    else:
        st.info(
            "Please click on the Update Chart button to load the candlestick chart for the selected stock",
            icon="ℹ️",
        )

with tab2:
    st.header("Initial Allocation of Capital using Monte Carlo Simulation")
    progress_text = "Calculating in progress. Please wait...."
    my_bar = st.progress(0, text=progress_text)
    d = {}
    for ticker in symbols:
        d[ticker] = df.query(f"Symbol=='{ticker}'")["Close"].tolist()
    price_df = pd.DataFrame(d)
    np.random.seed(42)
    num_iter = 10000

    sharpe_ratio_list = []
    portfolio_ret_list = []
    portfolio_vol_list = []
    w_list = []

    max_sharpe = 0
    max_sharpe_var = None
    max_sharpe_ret = None
    max_sharpe_w = None

    for i in tqdm(range(num_iter)):
        weights = np.random.random(len(symbols))
        weights /= np.sum(weights)
        portfolio = PortfolioStatistics(price_df, weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio.get_stats()
        my_bar.progress(i / 10000, text=progress_text)

        if sharpe_ratio > max_sharpe:
            max_sharpe = sharpe_ratio
            max_sharpe_vol = portfolio_volatility
            max_sharpe_ret = portfolio_return
            max_sharpe_w = weights

        portfolio_vol_list.append(portfolio_volatility)
        portfolio_ret_list.append(portfolio_return)
        w_list.append(weights)
        sharpe_ratio_list.append(sharpe_ratio)

    weights_d = {symbols[i]: max_sharpe_w[i] for i in range(len(symbols))}

    st.json(
        {
            "Portfolio with maximum Sharpe Ratio": max_sharpe,
            "Return": max_sharpe_ret,
            "Volatility": max_sharpe_vol,
            "Weights": max_sharpe_w,
        }
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        plot_returns_vs_volatility(
            portfolio_vol_list,
            portfolio_ret_list,
            sharpe_ratio_list,
            max_sharpe_vol,
            max_sharpe_ret,
        )
    with col2:
        plot_portfolio_weights(symbols, max_sharpe_w)

with tab3:

    supertrend_tester = SuperTrendVectorBacktester(
        df.query("Symbol=='META'"), float(initial_capital) * weights_d["META"]
    )
    supertrend_df = supertrend_tester.backtesting_flow()
    supertrend_sharpe = supertrend_df[
        "Total Strategy Portfolio Value"
    ].pct_change().mean() / (
        supertrend_df["Total Strategy Portfolio Value"].pct_change().std()
    )
    supertrend_sortino = supertrend_df[
        "Total Strategy Portfolio Value"
    ].pct_change().mean() / (
        supertrend_df["Total Strategy Portfolio Value"]
        .pct_change()
        .loc[supertrend_df["Total Strategy Portfolio Value"].pct_change() < 0]
        .std()
    )
    supertrend_profit_percentage = (
        (
            supertrend_df["Total Strategy Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["META"])
        )
        - 1
    ) * 100
    supertrend_buy_hold_percentage = (
        (
            supertrend_df["Buy and Hold Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["META"])
        )
        - 1
    ) * 100
    supertrend_max_drawdown = supertrend_df["Strategy Max Drawdown"].iloc[-1]
    supertrend_total_signa = supertrend_df.query("Signal!='Hold'").shape[0]

    st.header("Intuition behind the SuperTrend Strategy")
    annotated_text(
        "The SuperTrend strategy is a trend-following indicator",
        "that uses",
        ("Average True Range (ATR)", "", "#fea"),
        "to determine the direction of the trend. The ATR is a measure of market volatility, \
                and the SuperTrend indicator uses it to calculate the trend direction. \
                A buy signal is generated when the price crosses above the SuperTrend line, \
                indicating a bullish trend, while a sell signal occurs when the price crosses below the SuperTrend line, \
                signaling a bearish trend. This method aims to filter market noise and improve trade reliability by using \
                a volatility-based indicator to determine trend direction.",
    )
    with st.expander("Key Statistics"):
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        create_container(col1, "Sharpe Ratio", round(supertrend_sharpe, 2))
        create_container(col2, "Sortino Ratio", round(supertrend_sortino, 2))
        create_container(
            col3, "Strategy Profit %", f"{round(supertrend_profit_percentage,2)}%"
        )
        create_container(
            col4, "Buy&Hold Profit %", f"{round(supertrend_buy_hold_percentage,2)}%"
        )
        create_container(col5, "Max Drawdown", round(supertrend_max_drawdown, 2))
        create_container(col6, "Total No. of Signals", supertrend_total_signa)
    with st.expander("Processed data"):
        st.dataframe(
            supertrend_df, use_container_width=True, column_config=sma_column_config
        )
    with st.expander("Key visuals"):
        col1, col2 = st.columns([1, 1])
        with col1:
            plot_supertrend(supertrend_df, 3, 3)
            plot_drawdown_comparison(supertrend_df)
        with col2:
            plot_capital_changes(supertrend_df, "skyblue", "dodgerblue")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=supertrend_df["Date"],
                    y=supertrend_df["Total Strategy Portfolio Value"],
                    name="Strategy Capital Change",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df["Date"],
                    y=benchmark_df["Buy and Hold Portfolio Value"],
                    name="SP500 Buy and Hold Benchmark",
                )
            )
            fig.update_layout(
                title="Capital Change vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Capital",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with tab4:
    sma_tester = SMAVectorBacktester(
        df.query("Symbol=='AAPL'"), float(initial_capital) * weights_d["AAPL"]
    )
    sma_df = sma_tester.backtesting_flow()
    sma_sharpe = sma_df["Total Strategy Portfolio Value"].pct_change().mean() / (
        sma_df["Total Strategy Portfolio Value"].pct_change().std()
    )
    sma_sortino = sma_df["Total Strategy Portfolio Value"].pct_change().mean() / (
        sma_df["Total Strategy Portfolio Value"]
        .pct_change()
        .loc[sma_df["Total Strategy Portfolio Value"].pct_change() < 0]
        .std()
    )
    sma_profit_percentage = (
        (
            sma_df["Total Strategy Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["AAPL"])
        )
        - 1
    ) * 100
    sma_buy_hold_percentage = (
        (
            sma_df["Buy and Hold Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["AAPL"])
        )
        - 1
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
        create_container(col1, "Sharpe Ratio", round(sma_sharpe, 2))
        create_container(col2, "Sortino Ratio", round(sma_sortino, 2))
        create_container(
            col3, "Strategy Profit %", f"{round(sma_profit_percentage,2)}%"
        )
        create_container(
            col4, "Buy&Hold Profit %", f"{round(sma_buy_hold_percentage,2)}%"
        )
        create_container(col5, "Max Drawdown", round(sma_max_drawdown, 2))
        create_container(col6, "Total No. of Signals", sma_total_signa)
    with st.expander("Processed data"):
        st.dataframe(sma_df, use_container_width=True, column_config=sma_column_config)
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
                    y=sma_df["Total Strategy Portfolio Value"],
                    name="Strategy Capital Change",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df["Date"],
                    y=benchmark_df["Buy and Hold Portfolio Value"],
                    name="SP500 Buy and Hold Benchmark",
                )
            )
            fig.update_layout(
                title="Capital Change vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Capital",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # plot_profit_loss(sma_df, "crimson", "lightgrey")

with tab5:
    st.header("Intuition behind the Donchian Channel Strategy")
    annotated_text(
        "The Donchian Channel strategy, developed by Richard Donchian",
        "is a technical analysis tool used to identify market trends ",
        ("through the highest and lowest prices over a specific period.", "", "#fea"),
        "This period is defined by",
        ("upper_length and lower_length parameters", "", "#fea"),
        "which dictate the timeframe for the",
        ("highest high (upper channel) and lowest low (lower channel)", "", "#fea"),
        "respectively. Typically, traders use these channels to spot breakout points,"
        "going long when prices push above the upper channel and short when they drop "
        "below the lower channel, leveraging these parameters to fine-tune the strategy's sensitivity to market movements.",
    )
    dc_tester = DonchianChannelVectorBacktester(
        df.query("Symbol=='NFLX'"), float(initial_capital) * weights_d["NFLX"]
    )
    dc_df = dc_tester.backtesting_flow()
    dc_sharpe = dc_df["Total Strategy Portfolio Value"].pct_change().mean() / (
        dc_df["Total Strategy Portfolio Value"].pct_change().std()
    )
    dc_sortino = dc_df["Total Strategy Portfolio Value"].pct_change().mean() / (
        dc_df["Total Strategy Portfolio Value"]
        .pct_change()
        .loc[dc_df["Total Strategy Portfolio Value"].pct_change() < 0]
        .std()
    )
    dc_profit_percentage = (
        (
            dc_df["Total Strategy Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["NFLX"])
        )
        - 1
    ) * 100
    dc_buy_hold_percentage = (
        (
            dc_df["Buy and Hold Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["NFLX"])
        )
        - 1
    ) * 100
    dc_max_drawdown = dc_df["Strategy Max Drawdown"].iloc[-1]
    dc_total_signa = dc_df.query("Signal!='Hold'").shape[0]

    with st.expander("Key Statistics"):
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        create_container(col1, "Sharpe Ratio", round(dc_sharpe, 2))
        create_container(col2, "Sortino Ratio", round(dc_sortino, 2))
        create_container(col3, "Strategy Profit %", f"{round(dc_profit_percentage,2)}%")
        create_container(
            col4, "Buy&Hold Profit %", f"{round(dc_buy_hold_percentage,2)}%"
        )
        create_container(col5, "Max Drawdown", round(dc_max_drawdown, 2))
        create_container(col6, "Total No. of Signals", dc_total_signa)
    with st.expander("Processed data"):
        st.dataframe(dc_df, use_container_width=True, column_config=sma_column_config)
    with st.expander("Key visuals"):
        col1, col2 = st.columns([1, 1])
        with col1:
            plot_dc_trend(dc_df, 3, 3)
            # plot drawdown
            plot_drawdown_comparison(dc_df)

        with col2:
            plot_capital_changes(dc_df, "skyblue", "dodgerblue")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dc_df["Date"],
                    y=dc_df["Total Strategy Portfolio Value"],
                    name="Strategy Capital Change",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df["Date"],
                    y=benchmark_df["Buy and Hold Portfolio Value"],
                    name="SP500 Buy and Hold Benchmark",
                )
            )
            fig.update_layout(
                title="Capital Change vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Capital",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with tab6:
    ml_tester = MLClassifierVectorBacktester(
        df.query("Symbol=='AMZN'"), float(initial_capital) * weights_d["AMZN"]
    )
    ml_df, shap_df = ml_tester.backtesting_flow()
    ml_sharpe = ml_df["Total Strategy Portfolio Value"].pct_change().mean() / (
        ml_df["Total Strategy Portfolio Value"].pct_change().std()
    )
    ml_sortino = ml_df["Total Strategy Portfolio Value"].pct_change().mean() / (
        ml_df["Total Strategy Portfolio Value"]
        .pct_change()
        .loc[ml_df["Total Strategy Portfolio Value"].pct_change() < 0]
        .std()
    )
    ml_profit_percentage = (
        (
            ml_df["Total Strategy Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["AMZN"])
        )
        - 1
    ) * 100
    ml_buy_hold_percentage = (
        (
            ml_df["Buy and Hold Portfolio Value"].iloc[-1]
            / (float(initial_capital) * weights_d["AMZN"])
        )
        - 1
    ) * 100
    ml_max_drawdown = ml_df["Strategy Max Drawdown"].iloc[-1]
    ml_total_signa = ml_df.query("Signal!='Hold'").shape[0]

    st.header("Intuition behind the ML based Strategy")
    annotated_text(
        "The ML-based strategy uses",
        ("machine learning models", "", "#fea"),
        "to predict future price movements. The strategy uses historical price data to train the model, \
            which then generates buy and sell signals based on the predicted price movements. \
            This method aims to filter market noise and improve trade reliability by using \
            a machine learning model to predict future price movements.",
    )

    with st.expander("Key Statistics"):
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        create_container(col1, "Sharpe Ratio", round(ml_sharpe, 2))
        create_container(col2, "Sortino Ratio", round(ml_sortino, 2))
        create_container(col3, "Strategy Profit %", f"{round(ml_profit_percentage,2)}%")
        create_container(
            col4, "Buy&Hold Profit %", f"{round(ml_buy_hold_percentage,2)}%"
        )
        create_container(col5, "Max Drawdown", round(ml_max_drawdown, 2))
        create_container(col6, "Total No. of Signals", ml_total_signa)
    with st.expander("Processed data"):
        st.dataframe(ml_df, use_container_width=True, column_config=sma_column_config)
    with st.expander("Key visuals"):
        col1, col2 = st.columns([1, 1])
        with col1:
            plot_shap_bar_chart(shap_df)
            # plot drawdown
            plot_drawdown_comparison(ml_df)

        with col2:
            plot_capital_changes(ml_df, "skyblue", "dodgerblue")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ml_df["Date"],
                    y=ml_df["Total Strategy Portfolio Value"],
                    name="Strategy Capital Change",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df["Date"],
                    y=benchmark_df["Buy and Hold Portfolio Value"],
                    name="SP500 Buy and Hold Benchmark",
                )
            )
            fig.update_layout(
                title="Capital Change vs Benchmark",
                xaxis_title="Date",
                yaxis_title="Capital",
            )
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with tab7:
    st.header("Intuition behind the LSTM based Strategy")
    annotated_text(
        "The LSTM-based strategy uses",
        ("Long Short-Term Memory (LSTM) networks", "", "#fea"),
        "to predict future price movements. The strategy uses historical price data to train the model, \
            which then generates buy and sell signals based on the predicted price movements. \
            This method aims to filter market noise and improve trade reliability by using \
            a machine learning model to predict future price movements.",
    )
    with st.spinner("Model training in progress. Please wait...."):
        lstm_tester = LSTMVectorBacktester(
            df.query("Symbol=='GOOGL'"), float(initial_capital) * weights_d["GOOGL"]
        )
        lstm_df = lstm_tester.backtesting_flow()
        lstm_sharpe = lstm_df["Total Strategy Portfolio Value"].pct_change().mean() / (
            lstm_df["Total Strategy Portfolio Value"].pct_change().std()
        )
        lstm_sortino = lstm_df["Total Strategy Portfolio Value"].pct_change().mean() / (
            lstm_df["Total Strategy Portfolio Value"]
            .pct_change()
            .loc[lstm_df["Total Strategy Portfolio Value"].pct_change() < 0]
            .std()
        )
        lstm_profit_percentage = (
            (
                lstm_df["Total Strategy Portfolio Value"].iloc[-1]
                / (float(initial_capital) * weights_d["GOOGL"])
            )
            - 1
        ) * 100
        lstm_buy_hold_percentage = (
            (
                lstm_df["Buy and Hold Portfolio Value"].iloc[-1]
                / (float(initial_capital) * weights_d["GOOGL"])
            )
            - 1
        ) * 100
        lstm_max_drawdown = lstm_df["Strategy Max Drawdown"].iloc[-1]
        lstm_total_signa = lstm_df.query("Signal!='Hold'").shape[0]

        with st.expander("Key Statistics"):
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            create_container(col1, "Sharpe Ratio", round(lstm_sharpe, 2))
            create_container(col2, "Sortino Ratio", round(lstm_sortino, 2))
            create_container(
                col3, "Strategy Profit %", f"{round(lstm_profit_percentage,2)}%"
            )
            create_container(
                col4, "Buy&Hold Profit %", f"{round(lstm_buy_hold_percentage,2)}%"
            )
            create_container(col5, "Max Drawdown", round(lstm_max_drawdown, 2))
            create_container(col6, "Total No. of Signals", lstm_total_signa)

with tab8:

    portfolio_value = (
        supertrend_df["Total Strategy Portfolio Value"]
        + sma_df["Total Strategy Portfolio Value"]
        + dc_df["Total Strategy Portfolio Value"]
        + ml_df["Total Strategy Portfolio Value"]
        + lstm_df["Total Strategy Portfolio Value"]
    )

    buy_and_hold = (
        supertrend_df["Buy and Hold Portfolio Value"]
        + sma_df["Buy and Hold Portfolio Value"]
        + dc_df["Buy and Hold Portfolio Value"]
        + ml_df["Buy and Hold Portfolio Value"]
        + lstm_df["Buy and Hold Portfolio Value"]
    )

    final_df = pd.DataFrame(
        {
            "Date": supertrend_df["Date"],
            "Portfolio Value": portfolio_value,
            "Buy and Hold": buy_and_hold,
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=final_df["Date"],
            y=final_df["Portfolio Value"],
            name="Portfolio Value",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=final_df["Date"],
            y=final_df["Buy and Hold"],
            name="Buy and Hold Value",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=final_df["Date"],
            y=benchmark_df["Buy and Hold Portfolio Value"],
            name="SP500 Benchmark",
        )
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
