import yaml
import streamlit as st
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


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


import plotly.graph_objects as go
import streamlit as st


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
            y=data["Total Strategy Capital"],
            mode="lines",
            name="Strategy",
            line=dict(color=strategy_color, width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data.Date,
            y=data["Total Buy and Hold Capital"],
            mode="lines",
            name="Buy and Hold",
            line=dict(color=hold_color, width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data[data["Signal"] == "Buy"].Date,
            y=data[data["Signal"] == "Buy"]["Total Strategy Capital"],
            mode="markers",
            name="Buy",
            marker=dict(color="green", size=5),
        )
    )

    # Add markers for 'sell' signals
    fig.add_trace(
        go.Scatter(
            x=data[data["Signal"] == "Sell"].Date,
            y=data[data["Signal"] == "Sell"]["Total Strategy Capital"],
            mode="markers",
            name="Sell",
            marker=dict(color="red", size=5),
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
