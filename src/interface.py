import streamlit as st
from streamlit_lottie import st_lottie
from datetime import datetime
from util.utility_functions import fetch_data, plot_candlestick_with_indicators
from calculate_indicators import FinancialIndicators
from streamlit_extras.metric_cards import style_metric_cards
import plotly.graph_objects as go


# Global Variables
st.set_page_config(layout="wide")
st_lottie(
    "https://lottie.host/added42b-a2a9-42da-a297-744f34ec6533/ovGKQ7QQKu.json",
    height=400,
    width=800,
    speed=1,
    # key="initial",
)
st.title("Backtesting different trading strategies")

sidebar = st.sidebar
st.text("")
st.text("")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Exploratory Data Analysis",
        "MACD Strategy",
        "ML based Strategy",
        "Genetic Algorithm",
    ]
)


with sidebar:
    st.subheader("Backtesting Web App")
    st.divider()
    selected_stock = st.sidebar.text_input("Enter a valid stock ticker...", "GOOGL")
    st.text("")
    date_range = st.date_input(
        "Select date range for data extraction",
        [datetime(2015, 1, 1), datetime(2019, 12, 31)],
    )
    st.text("")
    button_clicked = st.sidebar.button("GO")


with tab1:

    if button_clicked:
        # spinner is not a must here because it loads really fast
        with st.spinner("Fetching data in progress..."):
            # Fetch the data from the API once the button is clicked
            df = fetch_data(selected_stock, date_range)
            df_copy = df.copy()
            calculator = FinancialIndicators(
                df_copy, lags=[1, 2, 3, 4, 5], windows=[5, 14, 30, 50, 100]
            )
            df_w_indicators = calculator.calculate_all_indicators()
            columns = ["Close", "RSI_14", "SMA_14", "EMA_14", "ATR_14", "MACD"]
            col_objects = st.columns(len(columns))

            for col, name in zip(col_objects, columns):
                col.metric(
                    name,
                    f"{round(df_w_indicators[name].iloc[-1], 2)}",
                    f"{round((df_w_indicators[name].iloc[-1] / df_w_indicators[name].iloc[-2] - 1)*100,2)}%",
                )

            style_metric_cards(border_left_color="#ADD8E6")
            col_1, col_2 = st.columns([1, 1])
            with col_1:
                st.dataframe(
                    df_copy.filter(items=["Open", "High", "Low", "Close", "Volume"]),
                    use_container_width=True,
                )
            with col_2:
                plot_candlestick_with_indicators(df_w_indicators, ["SMA_14", "EMA_14"])

        st.text("")
        st.dataframe(df_w_indicators.head(10), use_container_width=True)

with tab2:
    if button_clicked:
        st.header("MACD Strategy")
        st.markdown(
            """
The Moving Average Convergence Divergence (MACD) strategy is a technical analysis tool used to identify market trends and momentum. It consists of:

- **MACD Line**: The difference between the 12-period and 26-period Exponential Moving Averages.
- **Signal Line**: The 9-period EMA of the MACD Line.

**Key Signals**:
- **Crossovers**: Buy signal when the MACD Line crosses above the Signal Line. Sell signal when it crosses below.
- **Divergence**: Indicates potential trend reversal if the price moves opposite to MACD.
- **Overbought/Oversold**: Extreme MACD values may suggest reversal conditions.

It's favored for its simplicity and effectiveness in trending markets.
"""
        )
        df_MACD = df_w_indicators.copy()
        # Initialize the column
        df_MACD["Position"] = 0

        # Generate buy signals
        df_MACD.loc[df_MACD["MACD"] > df_MACD["Signal Line"], "Position"] = 1

        # Generate sell signals
        df_MACD.loc[df_MACD["MACD"] < df_MACD["Signal Line"], "Position"] = -1

        df_MACD["Strategy"] = df_MACD["Position"].shift(1) * df_MACD["Returns"]

        df_MACD["Cumulative Returns"] = df_MACD["Returns"].cumsum()
        df_MACD["Cumulative Strategy"] = df_MACD["Strategy"].cumsum()

        # plot a line chart using plotly
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_MACD.index,
                y=df_MACD["Cumulative Returns"],
                name="Cumulative Returns",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_MACD.index,
                y=df_MACD["Cumulative Strategy"],
                name="Cumulative Strategy",
            )
        )
        st.plotly_chart(fig, use_container_width=True)
