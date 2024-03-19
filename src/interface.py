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
    plot_value_strategy_vs_hold,
    create_stacked_bar_chart,
)
from util.st_column_config import sma_column_config
from util.performance_calculations import (
    calculate_annualized_sharpe_ratio,
    calculate_annualized_sortino_ratio,
    calculate_annualized_treynor_ratio,
    calculate_annualized_information_ratio,
    calculate_annualized_calmar_ratio,
)
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

benchmark_df["Market Returns"] = benchmark_df["Close"].pct_change()
benchmark_df["Buy and Hold Portfolio Value"] = (
    float(initial_capital) * (1 + benchmark_df["Market Returns"]).cumprod()
)

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
                super_weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["META"])
                )
                plot_capital_change_vs_benchmark(
                    supertrend_df, super_weighted_benchmark
                )
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
                sma_weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["AAPL"])
                )
                plot_capital_change_vs_benchmark(sma_df, sma_weighted_benchmark)
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
                dc_weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["NFLX"])
                )
                plot_capital_change_vs_benchmark(dc_df, dc_weighted_benchmark)
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
                ml_weighted_benchmark = calculate_sp500_portfolio_value(
                    benchmark_df, (float(initial_capital) * weights_d["AMZN"])
                )
                plot_capital_change_vs_benchmark(ml_df, ml_weighted_benchmark)
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
                    lstm_weighted_benchmark = calculate_sp500_portfolio_value(
                        benchmark_df, (float(initial_capital) * weights_d["GOOGL"])
                    )
                    plot_capital_change_vs_benchmark(lstm_df, lstm_weighted_benchmark)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )

with tab7:
    if button_clicked:

        portfolio_value = {
            "Date": supertrend_df["Date"],
            "SuperTrend": supertrend_df["Total Strategy Portfolio Value"],
            "SMA": sma_df["Total Strategy Portfolio Value"],
            "DC": dc_df["Total Strategy Portfolio Value"],
            "ML": ml_df["Total Strategy Portfolio Value"],
            "LSTM": lstm_df["Total Strategy Portfolio Value"],
            "Buy and Hold Cumulative Value": supertrend_df[
                "Buy and Hold Portfolio Value"
            ]
            + sma_df["Buy and Hold Portfolio Value"]
            + dc_df["Buy and Hold Portfolio Value"]
            + ml_df["Buy and Hold Portfolio Value"]
            + lstm_df["Buy and Hold Portfolio Value"],
        }

        portfolio_value_df = pd.DataFrame(portfolio_value)
        portfolio_value_df.index = supertrend_df["Date"]
        portfolio_value_df["Strategy Cumulative Value"] = (
            portfolio_value_df["SuperTrend"]
            + portfolio_value_df["SMA"]
            + portfolio_value_df["DC"]
            + portfolio_value_df["ML"]
            + portfolio_value_df["LSTM"]
        )

        portfolio_value_df["Strategy Returns"] = portfolio_value_df[
            "Strategy Cumulative Value"
        ].pct_change()
        portfolio_value_df["Buy and Hold Returns"] = portfolio_value_df[
            "Buy and Hold Cumulative Value"
        ].pct_change()
        portfolio_value_df["Strategy Cumulative Returns"] = (
            1 + portfolio_value_df["Strategy Returns"]
        ).cumprod()
        portfolio_value_df["Buy and Hold Cumulative Returns"] = (
            1 + portfolio_value_df["Buy and Hold Returns"]
        ).cumprod()

        portfolio_value_df["Strategy Drawdown"] = (
            portfolio_value_df["Strategy Cumulative Value"]
            - portfolio_value_df["Strategy Cumulative Value"].cummax()
        ) / portfolio_value_df["Strategy Cumulative Value"].cummax()
        portfolio_value_df["Buy and Hold Drawdown"] = (
            portfolio_value_df["Buy and Hold Cumulative Value"]
            - portfolio_value_df["Buy and Hold Cumulative Value"].cummax()
        ) / portfolio_value_df["Buy and Hold Cumulative Value"].cummax()

        strategy_max_drawdown = portfolio_value_df["Strategy Drawdown"].min()
        buy_and_hold_max_drawdown = portfolio_value_df["Buy and Hold Drawdown"].min()

        portfolio_value_df.dropna(inplace=True)

        with st.expander("Key Statistics"):
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            portfolio_sharpe = calculate_annualized_sharpe_ratio(
                portfolio_value_df["Strategy Returns"], 0
            )
            portfolio_sortino = calculate_annualized_sortino_ratio(
                portfolio_value_df["Strategy Returns"], 0
            )
            portfolio_treynor = calculate_annualized_treynor_ratio(
                portfolio_value_df["Strategy Returns"],
                portfolio_value_df["Buy and Hold Returns"],
                0,
            )
            portfolio_information = calculate_annualized_information_ratio(
                portfolio_value_df["Strategy Returns"],
                portfolio_value_df["Buy and Hold Returns"],
            )

            portfolio_calmar = calculate_annualized_calmar_ratio(
                portfolio_value_df["Strategy Returns"],
                strategy_max_drawdown,
            )
            create_container(col1, "Sharpe Ratio", round(portfolio_sharpe, 2))
            create_container(col2, "Sortino Ratio", round(portfolio_sortino, 2))
            create_container(col3, "Treynor Ratio", round(portfolio_treynor, 2))
            create_container(col4, "Information Ratio", round(portfolio_information, 2))
            create_container(
                col5, "Strategy Max Drawdown", round(strategy_max_drawdown, 2)
            )
            create_container(col6, "Calmar Ratio", round(portfolio_calmar, 2))

        with st.expander("Processed data"):
            st.dataframe(portfolio_value_df, use_container_width=True, hide_index=True)
        with st.expander("Key visuals"):
            col1, col2 = st.columns([1, 1])
            with col1:

                plot_value_strategy_vs_hold(portfolio_value_df)
                plot_drawdown_comparison(portfolio_value_df)
            with col2:
                create_stacked_bar_chart(portfolio_value_df)

                # Extract the year from the index
                portfolio_value_df["Date"] = pd.to_datetime(portfolio_value_df["Date"])
                portfolio_value_df["Year"] = portfolio_value_df["Date"].dt.year

                # Initialize lists to store results
                annualized_means_strategy, annualized_means_bnh = [], []
                annualized_volatilities_strategy, annualized_volatilities_bnh = [], []

                # Group by year
                for year, group in portfolio_value_df.groupby("Year"):
                    # Calculate mean daily return
                    mean_strategy_daily_return = np.mean(group["Strategy Returns"])
                    mean_bnh_daily_return = np.mean(group["Buy and Hold Returns"])

                    # Calculate standard deviation of daily return
                    std_dev_strategy_daily_return = np.std(group["Strategy Returns"])
                    std_dev_bnh_daily_return = np.std(group["Buy and Hold Returns"])

                    # Calculate annualized mean return
                    annualized_mean_strategy_return = (
                        1 + mean_strategy_daily_return
                    ) ** 252 - 1
                    annualized_mean_bnh_return = (1 + mean_bnh_daily_return) ** 252 - 1

                    # Calculate annualized standard deviation (volatility)
                    annualized_volatility_strategy = (
                        std_dev_strategy_daily_return * np.sqrt(252)
                    )
                    annualized_volatility_bnh = std_dev_bnh_daily_return * np.sqrt(252)

                    # Store results
                    annualized_means_strategy.append(annualized_mean_strategy_return)
                    annualized_means_bnh.append(annualized_mean_bnh_return)
                    annualized_volatilities_strategy.append(
                        annualized_volatility_strategy
                    )
                    annualized_volatilities_bnh.append(annualized_volatility_bnh)

                # Create a new dataframe with the calculated values
                df = pd.DataFrame(
                    {
                        "Year": portfolio_value_df["Year"].unique(),
                        "Annualized Mean Strategy Return": annualized_means_strategy,
                        "Annualized Mean BnH Return": annualized_means_bnh,
                        "Annualized Volatility Strategy": annualized_volatilities_strategy,
                        "Annualized Volatility BnH": annualized_volatilities_bnh,
                    }
                )

                # Create a dual y-axis plot
                fig = go.Figure()

                # Add traces for annualized mean returns
                fig.add_trace(
                    go.Bar(
                        x=df["Year"],
                        y=df["Annualized Mean Strategy Return"],
                        name="Annualized Mean Strategy Return",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["Annualized Mean BnH Return"],
                        name="Annualized Mean BnH Return",
                    )
                )

                # Create a second y-axis
                fig.update_layout(yaxis2=dict(overlaying="y", side="right"))

                # Add traces for annualized volatilities
                fig.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["Annualized Volatility Strategy"],
                        name="Annualized Volatility Strategy",
                        yaxis="y2",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["Year"],
                        y=df["Annualized Volatility BnH"],
                        name="Annualized Volatility BnH",
                        yaxis="y2",
                    )
                )

                fig.update_layout(
                    title="Annualized Mean Returns and Volatilities",
                    xaxis_title="Year",
                    yaxis_title="Annualized Mean Return",
                    yaxis2_title="Annualized Volatility",
                )

                # Display the figure
                st.plotly_chart(fig, use_container_width=True)

        # portfolio_mean = (
        #     weights_d["META"] * grouped_df["SuperTrend Returns"].mean()
        #     + weights_d["AAPL"] * grouped_df["SMA Returns"].mean()
        #     + weights_d["NFLX"] * grouped_df["DC Returns"].mean()
        #     + weights_d["AMZN"] * grouped_df["ML Returns"].mean()
        #     + weights_d["GOOGL"] * grouped_df["LSTM Returns"].mean()
        # )

        # cov_matrix = grouped_df.drop("Date", axis=1).cov()

        # portfolio_variance = np.dot(
        #     np.array(list(weights_d.values())).T,
        #     np.dot(cov_matrix, np.array(list(weights_d.values()))),
        # )

        # portfolio_volatility = np.sqrt(portfolio_variance)

        # portfolio_sharpe = portfolio_mean / portfolio_volatility

        # st.write(portfolio_sharpe)

        # st.dataframe(cov_matrix, use_container_width=True)
        # st.dataframe(return_df, use_container_width=True)
    else:
        st.info(
            "Please click on the Load Data button to start analyzing the selected stocks",
            icon="ℹ️",
        )
