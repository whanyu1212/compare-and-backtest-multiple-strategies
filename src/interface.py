import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm
from streamlit_lottie import st_lottie
from annotated_text import annotated_text
from datetime import datetime
from util.utility_functions import (
    parse_cfg,
    load_css,
    calculate_metrics,
    calculate_sp500_portfolio_value,
    create_container,
    plot_sma_trend,
    plot_dc_trend,
    plot_supertrend,
    plot_capital_changes,
    plot_drawdown_comparison,
    plot_shap_bar_chart,
    plot_returns_vs_volatility,
    plot_portfolio_weights,
    plot_stock_price_prediction,
    plot_capital_change_vs_benchmark,
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
from util.descriptions import (
    get_supertrend_description,
    get_sma_description,
    get_donchian_description,
    get_ml_strategy_description,
    get_lstm_strategy_description,
)
from monte_carlo import monte_carlo_optimize_portfolio

st.set_page_config(layout="wide")

symbols = parse_cfg("./config/parameters.yaml")["symbols"]


load_css("./src/style.css")


sidebar = st.sidebar
with sidebar:
    st.subheader("Settings")
    st.divider()
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
    button_clicked = st.sidebar.button("Load data")

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
    interval=interval,
)
benchmark_extractor = FinancialDataExtractor(
    symbols=["SPY"],
    start=date_range[0].strftime("%Y-%m-%d"),
    end=date_range[1].strftime("%Y-%m-%d"),
    interval=interval,
)

df = extractor.data.drop(["Dividends", "Stock Splits"], axis=1)

benchmark_df = benchmark_extractor.data.drop(
    ["Dividends", "Stock Splits"], axis=1
).reset_index()


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


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Portfolio Allocation",
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
        st.header("Generating Initial Portfolio Weights")
        with st.spinner("Running Monte Carlo Simulation..."):

            d = {}
            for ticker in symbols:
                d[ticker] = df.query(f"Symbol=='{ticker}'")["Close"].tolist()
            price_df = pd.DataFrame(d)

            (
                max_sharpe,
                max_sharpe_vol,
                max_sharpe_ret,
                max_sharpe_w,
                portfolio_vol_list,
                portfolio_ret_list,
                w_list,
                sharpe_ratio_list,
            ) = monte_carlo_optimize_portfolio(price_df, symbols)

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
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab2:
    if button_clicked:
        supertrend_tester = SuperTrendVectorBacktester(
            df.query("Symbol=='META'"), float(initial_capital) * weights_d["META"]
        )
        supertrend_df = supertrend_tester.backtesting_flow()
        (
            supertrend_sharpe,
            supertrend_sortino,
            supertrend_profit_percentage,
            supertrend_buy_hold_percentage,
            supertrend_max_drawdown,
            supertrend_total_signal,
        ) = calculate_metrics(supertrend_df, initial_capital, "META", weights_d)

        st.header("Intuition behind the SuperTrend Strategy")
        annotated_text(*get_supertrend_description())
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
            create_container(col6, "Total No. of Signals", supertrend_total_signal)
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
                weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["META"])
                )
                plot_capital_change_vs_benchmark(supertrend_df, weighted_benchmark)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab3:
    if button_clicked:
        sma_tester = SMAVectorBacktester(
            df.query("Symbol=='AAPL'"), float(initial_capital) * weights_d["AAPL"]
        )
        sma_df = sma_tester.backtesting_flow()
        (
            sma_sharpe,
            sma_sortino,
            sma_profit_percentage,
            sma_buy_hold_percentage,
            sma_max_drawdown,
            sma_total_signal,
        ) = calculate_metrics(sma_df, initial_capital, "AAPL", weights_d)

        st.header("Intuition behind the Triple SMA Crossover Strategy")
        annotated_text(*get_sma_description())
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
            create_container(col6, "Total No. of Signals", sma_total_signal)
        with st.expander("Processed data"):
            st.dataframe(
                sma_df, use_container_width=True, column_config=sma_column_config
            )
        with st.expander("Key visuals"):
            col1, col2 = st.columns([1, 1])
            with col1:
                plot_sma_trend(sma_df, 3, 3)

                plot_drawdown_comparison(sma_df)

            with col2:
                plot_capital_changes(sma_df, "skyblue", "dodgerblue")
                weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["AAPL"])
                )
                plot_capital_change_vs_benchmark(sma_df, weighted_benchmark)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab4:
    if button_clicked:
        st.header("Intuition behind the Donchian Channel Strategy")
        annotated_text(*get_donchian_description())
        dc_tester = DonchianChannelVectorBacktester(
            df.query("Symbol=='NFLX'"), float(initial_capital) * weights_d["NFLX"]
        )
        dc_df = dc_tester.backtesting_flow()
        (
            dc_sharpe,
            dc_sortino,
            dc_profit_percentage,
            dc_buy_hold_percentage,
            dc_max_drawdown,
            dc_total_signal,
        ) = calculate_metrics(dc_df, initial_capital, "NFLX", weights_d)

        with st.expander("Key Statistics"):
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            create_container(col1, "Sharpe Ratio", round(dc_sharpe, 2))
            create_container(col2, "Sortino Ratio", round(dc_sortino, 2))
            create_container(
                col3, "Strategy Profit %", f"{round(dc_profit_percentage,2)}%"
            )
            create_container(
                col4, "Buy&Hold Profit %", f"{round(dc_buy_hold_percentage,2)}%"
            )
            create_container(col5, "Max Drawdown", round(dc_max_drawdown, 2))
            create_container(col6, "Total No. of Signals", dc_total_signal)
        with st.expander("Processed data"):
            st.dataframe(
                dc_df, use_container_width=True, column_config=sma_column_config
            )
        with st.expander("Key visuals"):
            col1, col2 = st.columns([1, 1])
            with col1:
                plot_dc_trend(dc_df, 3, 3)

                plot_drawdown_comparison(dc_df)

            with col2:
                plot_capital_changes(dc_df, "skyblue", "dodgerblue")
                weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["NFLX"])
                )
                plot_capital_change_vs_benchmark(dc_df, weighted_benchmark)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab5:
    if button_clicked:
        ml_tester = MLClassifierVectorBacktester(
            df.query("Symbol=='AMZN'"), float(initial_capital) * weights_d["AMZN"]
        )
        ml_df, shap_df = ml_tester.backtesting_flow()
        (
            ml_sharpe,
            ml_sortino,
            ml_profit_percentage,
            ml_buy_hold_percentage,
            ml_max_drawdown,
            ml_total_signal,
        ) = calculate_metrics(ml_df, initial_capital, "AMZN", weights_d)

        st.header("Intuition behind the ML based Strategy")
        annotated_text(*get_ml_strategy_description())

        with st.expander("Key Statistics"):
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            create_container(col1, "Sharpe Ratio", round(ml_sharpe, 2))
            create_container(col2, "Sortino Ratio", round(ml_sortino, 2))
            create_container(
                col3, "Strategy Profit %", f"{round(ml_profit_percentage,2)}%"
            )
            create_container(
                col4, "Buy&Hold Profit %", f"{round(ml_buy_hold_percentage,2)}%"
            )
            create_container(col5, "Max Drawdown", round(ml_max_drawdown, 2))
            create_container(col6, "Total No. of Signals", ml_total_signal)
        with st.expander("Processed data"):
            st.dataframe(
                ml_df, use_container_width=True, column_config=sma_column_config
            )
        with st.expander("Key visuals"):
            col1, col2 = st.columns([1, 1])
            with col1:
                plot_shap_bar_chart(shap_df)

                plot_drawdown_comparison(ml_df)

            with col2:
                plot_capital_changes(ml_df, "skyblue", "dodgerblue")
                weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["AMZN"])
                )
                plot_capital_change_vs_benchmark(ml_df, weighted_benchmark)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab6:
    if button_clicked:
        st.header("Intuition behind the LSTM based Strategy")
        annotated_text(*get_lstm_strategy_description())
        with st.spinner("Model training in progress. Please wait...."):
            lstm_tester = LSTMVectorBacktester(
                df.query("Symbol=='GOOGL'"), float(initial_capital) * weights_d["GOOGL"]
            )
            lstm_df = lstm_tester.backtesting_flow()
            (
                lstm_sharpe,
                lstm_sortino,
                lstm_profit_percentage,
                lstm_buy_hold_percentage,
                lstm_max_drawdown,
                lstm_total_signal,
            ) = calculate_metrics(lstm_df, initial_capital, "GOOGL", weights_d)

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
                create_container(col6, "Total No. of Signals", lstm_total_signal)
            with st.expander("Processed data"):
                st.dataframe(
                    lstm_df, use_container_width=True, column_config=sma_column_config
                )
            with st.expander("Key visuals"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    plot_stock_price_prediction(
                        lstm_df["Close"], lstm_df["Predicted Close"]
                    )

                    plot_drawdown_comparison(lstm_df)

                with col2:
                    plot_capital_changes(lstm_df, "skyblue", "dodgerblue")
                    weighted_benchmark = calculate_sp500_portfolio_value(
                        benchmark_df, (float(initial_capital) * weights_d["GOOGL"])
                    )
                plot_capital_change_vs_benchmark(lstm_df, weighted_benchmark)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab7:
    if button_clicked:

        sma_returns = sma_df["Total Strategy Portfolio Value"].pct_change()
        supertrend_returns = supertrend_df[
            "Total Strategy Portfolio Value"
        ].pct_change()
        dc_returns = dc_df["Total Strategy Portfolio Value"].pct_change()
        ml_returns = ml_df["Total Strategy Portfolio Value"].pct_change()
        lstm_returns = lstm_df["Total Strategy Portfolio Value"].pct_change()

        return_df = pd.DataFrame(
            {
                "Date": sma_df["Date"],
                "SuperTrend Returns": supertrend_returns,
                "SMA Returns": sma_returns,
                "DC Returns": dc_returns,
                "ML Returns": ml_returns,
                "LSTM Returns": lstm_returns,
            }
        ).dropna()

        sma_value = sma_df["Total Strategy Portfolio Value"]
        sma_buy_and_hold_value = sma_df["Buy and Hold Portfolio Value"]
        supertrend_value = supertrend_df["Total Strategy Portfolio Value"]
        supertrend_buy_and_hold_value = supertrend_df["Buy and Hold Portfolio Value"]
        dc_value = dc_df["Total Strategy Portfolio Value"]
        dc_buy_and_hold_value = dc_df["Buy and Hold Portfolio Value"]
        ml_value = ml_df["Total Strategy Portfolio Value"]
        ml_buy_and_hold_value = ml_df["Buy and Hold Portfolio Value"]
        lstm_value = lstm_df["Total Strategy Portfolio Value"]
        lstm_buy_and_hold_value = lstm_df["Buy and Hold Portfolio Value"]

        value_df = pd.DataFrame(
            {
                "Date": sma_df["Date"],
                "SuperTrend Value": supertrend_value,
                "SMA Value": sma_value,
                "DC Value": dc_value,
                "ML Value": ml_value,
                "LSTM Value": lstm_value,
                "Strategy Cumulative Value": supertrend_value
                + sma_value
                + dc_value
                + ml_value
                + lstm_value,
                "Buy and Hold Cumulative Value": supertrend_buy_and_hold_value
                + sma_buy_and_hold_value
                + dc_buy_and_hold_value
                + ml_buy_and_hold_value
                + lstm_buy_and_hold_value,
                "SP500 Benchmark": benchmark_df["Buy and Hold Portfolio Value"],
            }
        ).dropna()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=value_df["Date"],
                y=value_df["Strategy Cumulative Value"],
                name="Strategy Cumulative Value",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=value_df["Date"],
                y=value_df["Buy and Hold Cumulative Value"],
                name="Buy and Hold Cumulative Value",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=value_df["Date"],
                y=value_df["SP500 Benchmark"],
                name="SP500 Buy and Hold Benchmark",
            )
        )

        fig.update_layout(
            title="Strategy Cumulative Value vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Capital",
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        return_df["Date"] = pd.to_datetime(return_df["Date"])
        grouped_df = (
            return_df.groupby(pd.Grouper(key="Date", freq="Y")).sum().reset_index()
        )

        st.markdown("## Annual Returns")
        st.dataframe(grouped_df, use_container_width=True)

        # st.write(return_df.drop("Date", axis=1).mean() * 252)

        portfolio_mean = (
            weights_d["META"] * grouped_df["SuperTrend Returns"].mean()
            + weights_d["AAPL"] * grouped_df["SMA Returns"].mean()
            + weights_d["NFLX"] * grouped_df["DC Returns"].mean()
            + weights_d["AMZN"] * grouped_df["ML Returns"].mean()
            + weights_d["GOOGL"] * grouped_df["LSTM Returns"].mean()
        )

        cov_matrix = grouped_df.drop("Date", axis=1).cov()

        portfolio_variance = np.dot(
            np.array(list(weights_d.values())).T,
            np.dot(cov_matrix, np.array(list(weights_d.values()))),
        )

        portfolio_volatility = np.sqrt(portfolio_variance)

        portfolio_sharpe = portfolio_mean / portfolio_volatility

        st.write(portfolio_sharpe)

        # st.dataframe(cov_matrix, use_container_width=True)
        # st.dataframe(return_df, use_container_width=True)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )
