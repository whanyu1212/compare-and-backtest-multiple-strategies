import streamlit as st

sma_column_config = {
    "Close": st.column_config.NumberColumn(
        "Closing Price",
        help="Daily Closing Price",
        format="$%.2f",
    ),
    "SMA1": st.column_config.NumberColumn(
        "SMA1",
        help="Short Term Moving Average",
        format="$%.2f",
    ),
    "SMA2": st.column_config.NumberColumn(
        "SMA2",
        help="Medium Term Moving Average",
        format="$%.2f",
    ),
    "SMA3": st.column_config.NumberColumn(
        "SMA3",
        help="Long Term Moving Average",
        format="$%.2f",
    ),
    "Price Returns": st.column_config.NumberColumn(
        "Price Returns",
        help="Return based on the closing price",
        format="%.2f",
    ),
    "Balance": st.column_config.NumberColumn(
        "Balance",
        help="Balance",
        format="$%.2f",
    ),
    "Realized PnL": st.column_config.NumberColumn(
        "Realized PnL",
        help="Realized Profit and Loss",
        format="$%.2f",
    ),
    "Unrealized PnL": st.column_config.NumberColumn(
        "Unrealized PnL",
        help="Unrealized Profit and Loss",
        format="$%.2f",
    ),
    "Total Strategy Capital": st.column_config.NumberColumn(
        "Cumulative Capital (Strategy)",
        help="Cumulative Capital (Strategy)",
        format="$%.2f",
    ),
    "Total Buy and Hold Capital": st.column_config.NumberColumn(
        "Cumulative Capital (B&H)",
        help="Cumulative Capital (B&H)",
        format="$%.2f",
    ),
}
