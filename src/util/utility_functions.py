import yaml
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
    trace_close = go.Scatter(
        x=data["Date"][:100],
        y=data["Close"][:100],
        mode="lines",
        name="AAPL",
        line=dict(width=close_width),
    )

    trace_sma1 = go.Scatter(
        x=data["Date"][:100],
        y=data["SMA1"][:100],
        mode="lines",
        name="SMA Short",
        line=dict(dash="dot", width=sma_width),
    )

    trace_sma2 = go.Scatter(
        x=data["Date"][:100],
        y=data["SMA2"][:100],
        mode="lines",
        name="SMA Medium",
        line=dict(dash="dot", width=sma_width),
    )

    trace_sma3 = go.Scatter(
        x=data["Date"][:100],
        y=data["SMA3"][:100],
        mode="lines",
        name="SMA Long",
        line=dict(dash="dot", width=sma_width),
    )

    layout = go.Layout(
        title="Trend with SMA Lines (first 100 points)",
        xaxis_title="Date",
        yaxis_title="Price",
    )

    fig = go.Figure(
        data=[trace_close, trace_sma1, trace_sma2, trace_sma3], layout=layout
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_capital_changes(data, strategy_color, hold_color):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.Date,
            y=data["Total Strategy Portfolio Value"],
            mode="lines",
            name="Strategy",
            line=dict(color=strategy_color, width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.Date,
            y=data["Buy and Hold Portfolio Value"],
            mode="lines",
            name="Buy and Hold",
            line=dict(color=hold_color, width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data[data["Signal"] == "Buy"].Date,
            y=data[data["Signal"] == "Buy"]["Total Strategy Portfolio Value"],
            mode="markers",
            name="Buy",
            marker=dict(color="green", size=8, symbol="triangle-up"),
        )
    )

    # Add markers for 'sell' signals
    fig.add_trace(
        go.Scatter(
            x=data[data["Signal"] == "Sell"].Date,
            y=data[data["Signal"] == "Sell"]["Total Strategy Portfolio Value"],
            mode="markers",
            name="Sell",
            marker=dict(color="red", size=8, symbol="triangle-down"),
        )
    )

    fig.update_layout(
        title="Capital Change Comparison",
        xaxis_title="Date",
        yaxis_title="Capital",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_profit_loss(data, realized_color, unrealized_color):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=data.Date,
            y=data["Realized PnL"],
            name="Realized Profit & Loss",
            marker_color=realized_color,
        )
    )
    fig.add_trace(
        go.Bar(
            x=data.Date,
            y=data["Unrealized PnL"],
            name="Unrealized Profit & Loss",
            marker_color=unrealized_color,
        )
    )

    fig.update_layout(
        title="Realized / Unrealized Profit & Loss over time",
        xaxis_title="Date",
        yaxis_title="Profit / Loss",
        barmode="group",
        xaxis_tickangle=-45,
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
    )

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_drawdown_comparison(sma_df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sma_df["Date"],
            y=sma_df["Strategy Drawdown"],
            name="Strategy Drawdown",
            line=dict(color="skyblue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sma_df["Date"],
            y=sma_df["Buy and Hold Drawdown"],
            name="Buy and Hold Drawdown",
            line=dict(color="dodgerblue"),
        )
    )
    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown",
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_dc_trend(data, close_width, dc_width):
    trace_close = go.Scatter(
        x=data["Date"][:100],
        y=data["Close"][:100],
        mode="lines",
        name="GOOGL",
        line=dict(width=close_width),
    )

    trace_dc1 = go.Scatter(
        x=data["Date"][:100],
        y=data["dcl"][:100],
        mode="lines",
        name="dcl",
        line=dict(dash="dot", width=dc_width),
    )

    trace_dc2 = go.Scatter(
        x=data["Date"][:100],
        y=data["dcm"][:100],
        mode="lines",
        name="dcm",
        line=dict(dash="dot", width=dc_width),
    )

    trace_dc3 = go.Scatter(
        x=data["Date"][:100],
        y=data["dcu"][:100],
        mode="lines",
        name="dcu",
        line=dict(dash="dot", width=dc_width),
    )

    layout = go.Layout(
        title="Trend with DC Lines (first 100 points)",
        xaxis_title="Date",
        yaxis_title="Price",
    )

    fig = go.Figure(data=[trace_close, trace_dc1, trace_dc2, trace_dc3], layout=layout)

    st.plotly_chart(fig, use_container_width=True, theme="streamlit")


def plot_supertrend(data, close_width, supertrend_width):
    trace_close = go.Scatter(
        x=data["Date"][:100],
        y=data["Close"][:100],
        mode="lines",
        name="META",
        line=dict(width=close_width),
    )

    trace_supertrend = go.Scatter(
        x=data["Date"][:100],
        y=data["Supertrend"][:100],
        mode="lines",
        name="Supertrend",
        line=dict(dash="dot", width=supertrend_width),
    )

    layout = go.Layout(
        title="Trend with Supertrend (first 100 points)",
        xaxis_title="Date",
        yaxis_title="Price",
    )

    fig = go.Figure(data=[trace_close, trace_supertrend], layout=layout)

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


import plotly.graph_objects as go
import streamlit as st


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
