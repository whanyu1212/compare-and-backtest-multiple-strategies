import yaml
import math
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fetch_data import FinancialDataExtractor


def parse_cfg(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def fetch_data(selected_stock, date_range, interval="1d"):
    extractor = FinancialDataExtractor(
        symbol=selected_stock,
        start=date_range[0].strftime("%Y-%m-%d"),
        end=date_range[1].strftime("%Y-%m-%d"),
        interval=interval,
    )

    df = extractor.data

    return df


def calculate_metrics(
    df,
    initial_capital,
    symbol,
    weights,
    strategy_column="Total Strategy Portfolio Value",
    buy_hold_column="Buy and Hold Portfolio Value",
):
    pct_change = df[strategy_column].pct_change()
    mean_pct_change = pct_change.mean()
    std_pct_change = pct_change.std()
    negative_std_pct_change = pct_change.loc[pct_change < 0].std()

    sharpe = mean_pct_change * np.sqrt(252) / std_pct_change
    sortino = mean_pct_change * np.sqrt(252) / negative_std_pct_change
    profit_percentage = (
        (df[strategy_column].iloc[-1] / (float(initial_capital) * weights[symbol])) - 1
    ) * 100
    buy_hold_percentage = (
        (df[buy_hold_column].iloc[-1] / (float(initial_capital) * weights[symbol])) - 1
    ) * 100
    max_drawdown = df["Strategy Max Drawdown"].iloc[-1]
    total_signal = df.query("Signal!='Hold'").shape[0]

    return (
        sharpe,
        sortino,
        profit_percentage,
        buy_hold_percentage,
        max_drawdown,
        total_signal,
    )


def calculate_sp500_portfolio_value(benchmark_df, initial_capital):
    benchmark_df["Market Returns"] = benchmark_df["Close"].pct_change()
    left_over = (
        float(initial_capital)
        - math.floor(float(initial_capital) / benchmark_df.loc[0, "Close"])
        * benchmark_df.loc[0, "Close"]
    )
    benchmark_df["No of shares"] = math.floor(
        float(initial_capital) / benchmark_df.loc[0, "Close"]
    )
    benchmark_df["Buy and Hold Portfolio Value"] = (
        benchmark_df["No of shares"] * benchmark_df["Close"] + left_over
    )
    benchmark_df["Buy and Hold Portfolio Value"].fillna(
        float(initial_capital), inplace=True
    )
    return benchmark_df


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def create_container(column, title, value):
    with column:
        with st.container(height=100):
            st.markdown(
                f"<h2 style='text-align: center; font-family: Space Grotesk;font-size: 14px;'>{title}: <br> {value} </h2>",
                unsafe_allow_html=True,
            )


def plot_candlestick_with_indicators(df, indicators):
    # Create the candlestick chart
    candlestick = go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        showlegend=False,
    )

    # Create the indicator charts
    indicator_charts = []
    for indicator in indicators:
        indicator_chart = go.Scatter(
            x=df.index,
            y=df[indicator],
            name=indicator,
            line=dict(color="orange") if "EMA" in indicator else None,
        )
        indicator_charts.append(indicator_chart)

    # Combine the charts
    fig = go.Figure(data=[candlestick] + indicator_charts)

    # Update the layout
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

    # Display the chart
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_sma_trend(data, close_width, sma_width):
    traces = []
    labels = ["AAPL", "SMA Short", "SMA Medium", "SMA Long"]
    columns = ["Close", "SMA1", "SMA2", "SMA3"]
    widths = [close_width, sma_width, sma_width, sma_width]

    for label, column, width in zip(labels, columns, widths):
        trace = go.Scatter(
            x=data["Date"][-100:],
            y=data[column][-100:],
            mode="lines",
            name=label,
            line=dict(dash="dot" if "SMA" in label else None, width=width),
        )
        traces.append(trace)

    layout = go.Layout(
        title="Trend with SMA Lines (last 100 days)",
        xaxis_title="Date",
        yaxis_title="Price",
    )

    fig = go.Figure(data=traces, layout=layout)

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def create_trace(x, y, mode, name, line=None, marker=None):
    return go.Scatter(x=x, y=y, mode=mode, name=name, line=line, marker=marker)


def plot_capital_changes(data, strategy_color, hold_color):
    fig = go.Figure()

    traces = [
        (
            data.Date,
            data["Total Strategy Portfolio Value"],
            "lines",
            "Strategy",
            dict(color=strategy_color, width=3),
            None,
        ),
        (
            data.Date,
            data["Buy and Hold Portfolio Value"],
            "lines",
            "Buy and Hold",
            dict(color=hold_color, width=3),
            None,
        ),
        (
            data[data["Signal"] == "Buy"].Date,
            data[data["Signal"] == "Buy"]["Total Strategy Portfolio Value"],
            "markers",
            "Buy",
            None,
            dict(color="green", size=8, symbol="triangle-up", opacity=0.5),
        ),
        (
            data[data["Signal"] == "Sell"].Date,
            data[data["Signal"] == "Sell"]["Total Strategy Portfolio Value"],
            "markers",
            "Sell",
            None,
            dict(color="red", size=8, symbol="triangle-down", opacity=0.5),
        ),
    ]

    for trace in traces:
        fig.add_trace(create_trace(*trace))

    fig.update_layout(
        title="Capital Change Comparison",
        xaxis_title="Date",
        yaxis_title="Capital",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_drawdown_comparison(df):
    fig = go.Figure()

    traces = [
        (
            df["Date"],
            df["Strategy Drawdown"],
            "lines",
            "Strategy Drawdown",
            dict(color="skyblue"),
            None,
        ),
        (
            df["Date"],
            df["Buy and Hold Drawdown"],
            "lines",
            "Buy and Hold Drawdown",
            dict(color="dodgerblue"),
            None,
        ),
    ]

    for trace in traces:
        fig.add_trace(create_trace(*trace))

    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_dc_trend(data, close_width, dc_width):
    traces = [
        (
            data["Date"][-100:],
            data["Close"][-100:],
            "lines",
            "NFLX",
            dict(width=close_width),
            None,
        ),
        (
            data["Date"][-100:],
            data["dcl"][-100:],
            "lines",
            "dcl",
            dict(dash="dot", width=dc_width),
            None,
        ),
        (
            data["Date"][-100:],
            data["dcm"][-100:],
            "lines",
            "dcm",
            dict(dash="dot", width=dc_width),
            None,
        ),
        (
            data["Date"][-100:],
            data["dcu"][-100:],
            "lines",
            "dcu",
            dict(dash="dot", width=dc_width),
            None,
        ),
    ]

    fig = go.Figure(data=[create_trace(*trace) for trace in traces])

    fig.update_layout(
        title="Trend with DC Lines (last 100 days)",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_supertrend(data, close_width, supertrend_width):
    traces = [
        (
            data["Date"][-100:],
            data["Close"][-100:],
            "lines",
            "META",
            dict(width=close_width),
            None,
        ),
        (
            data["Date"][-100:],
            data["Supertrend"][-100:],
            "lines",
            "Supertrend",
            dict(dash="dot", width=supertrend_width),
            None,
        ),
    ]

    fig = go.Figure(data=[create_trace(*trace) for trace in traces])

    fig.update_layout(
        title="Trend with Supertrend (last 100 days)",
        xaxis_title="Date",
        yaxis_title="Price",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_stock_price_prediction(y_train, y_pred):
    fig = go.Figure()

    traces = [
        (
            list(range(len(y_train))),
            y_train,
            "lines",
            "Actual",
            dict(color="blue", width=3),
            None,
        ),
        (
            list(range(len(y_pred))),
            y_pred,
            "lines",
            "Predicted",
            dict(color="orange", width=1.5),
            None,
        ),
    ]

    for trace in traces:
        fig.add_trace(create_trace(*trace))

    fig.update_layout(
        title="GOOGL Stock Price Prediction",
        xaxis_title="Days",
        yaxis_title="Stock Price",
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_shap_bar_chart(importance_df):
    fig = px.bar(
        importance_df,
        x="shap_values",
        y="col_name",
        orientation="h",
        title="Feature Importance based on SHAP values",
        color="shap_values",
        labels={"shap_values": "SHAP Importance", "col_name": "features"},
        color_continuous_scale="Blues",
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_returns_vs_volatility(
    portfolio_vol_list,
    portfolio_ret_list,
    sharpe_ratio_list,
    max_sharpe_vol,
    max_sharpe_ret,
):
    fig = go.Figure()

    # Add the scatter plot
    fig.add_trace(
        go.Scatter(
            x=portfolio_vol_list,
            y=portfolio_ret_list,
            mode="markers",
            marker=dict(
                size=8,
                color=sharpe_ratio_list,  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                colorbar=dict(title="Sharpe Ratio"),
                showscale=False,
            ),
            name="Portfolios",
        )
    )

    # Add a marker for the point with the highest Sharpe ratio
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe_vol],
            y=[max_sharpe_ret],
            mode="markers",
            marker=dict(size=10, color="Red", symbol="star"),
            name="Max Sharpe Ratio",
        )
    )

    # Add labels and title
    fig.update_layout(
        title="Returns vs Volatility", xaxis_title="Volatility", yaxis_title="Returns"
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_portfolio_weights(symbols, max_sharpe_w):
    values = [i * 100 for i in list(max_sharpe_w)]  # Convert weights to percentages

    fig = go.Figure(data=[go.Pie(labels=symbols, values=values, hole=0.5)])

    fig.update_layout(
        title_text="Portfolio Weights in % - MAANG",
        annotations=[
            dict(
                text="",
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False,
            )
        ],
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_capital_change_vs_benchmark(strategy_df, benchmark_df):
    fig = go.Figure()

    traces = [
        (
            strategy_df["Date"],
            strategy_df["Total Strategy Portfolio Value"],
            "lines",
            "Strategy Capital Change",
            None,
            None,
        ),
        (
            benchmark_df["Date"],
            benchmark_df["Buy and Hold Portfolio Value"],
            "lines",
            "SP500 Benchmark",
            None,
            None,
        ),
    ]

    for trace in traces:
        fig.add_trace(create_trace(*trace))

    fig.update_layout(
        title="Capital Change vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Capital",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_value_strategy_vs_hold(value_df):
    fig = go.Figure()
    fig.add_trace(
        create_trace(
            value_df["Date"],
            value_df["Strategy Cumulative Value"],
            "lines",
            "Strategy Cumulative Value",
        )
    )
    fig.add_trace(
        create_trace(
            value_df["Date"],
            value_df["Buy and Hold Cumulative Value"],
            "lines",
            "Buy and Hold Cumulative Value",
        )
    )

    fig.update_layout(
        title="Strategy Cumulative Value vs Buy and Hold Cumulative Value",
        xaxis_title="Date",
        yaxis_title="Capital",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_return_strategy_vs_hold(value_df):
    fig = go.Figure()
    fig.add_trace(
        create_trace(
            value_df["Date"],
            value_df["Strategy Cumulative Returns"],
            "lines",
            "Strategy Cumulative Returns",
        )
    )
    fig.add_trace(
        create_trace(
            value_df["Date"],
            value_df["Buy and Hold Cumulative Returns"],
            "lines",
            "lines" "Buy and Hold Cumulative Returns",
        )
    )

    fig.update_layout(
        title="Strategy Cumulative Return vs Buy and Hold Cumulative Return",
        xaxis_title="Date",
        yaxis_title="Return",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def create_stacked_bar_chart(portfolio_value):
    fig = go.Figure(
        data=[
            go.Bar(
                name="SuperTrend",
                x=portfolio_value["Date"],
                y=portfolio_value["SuperTrend"],
            ),
            go.Bar(name="SMA", x=portfolio_value["Date"], y=portfolio_value["SMA"]),
            go.Bar(name="DC", x=portfolio_value["Date"], y=portfolio_value["DC"]),
            go.Bar(name="ML", x=portfolio_value["Date"], y=portfolio_value["ML"]),
            go.Bar(name="LSTM", x=portfolio_value["Date"], y=portfolio_value["LSTM"]),
        ]
    )

    # Change the bar mode
    fig.update_layout(barmode="stack", title="Return breakdown by strategy")

    # Display the figure
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_profit_loss_distribution(portfolio_value_df):
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=portfolio_value_df["Strategy Returns"],
            name="Strategy Returns",
            opacity=0.75,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=portfolio_value_df["Buy and Hold Returns"],
            name="Buy and Hold Returns",
            opacity=0.75,
        )
    )

    fig.update_layout(
        barmode="overlay",
        title_text="Profit/Loss Distribution",
        xaxis_title_text="Returns",
        yaxis_title_text="Frequency",
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
