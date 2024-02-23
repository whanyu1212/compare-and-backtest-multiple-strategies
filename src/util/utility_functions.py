import yaml
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
