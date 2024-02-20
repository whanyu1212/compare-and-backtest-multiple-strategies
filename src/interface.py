import streamlit as st
from streamlit_lottie import st_lottie

# Global Variables
st.set_page_config(layout="wide")
st_lottie(
    "https://lottie.host/added42b-a2a9-42da-a297-744f34ec6533/ovGKQ7QQKu.json",
    height=300,
    width=600,
    speed=1,
    # key="initial",
)
st.title("Comparing Backtesting Results of Different Strategies")
