import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from src.util.utility_functions import parse_cfg


class FinancialDataExtractor:
    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
        max_retries: int = 3,
        delay: int = 1,
    ):
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Symbol must be a non-empty string.")

        valid_intervals = parse_cfg("./config/parameters.yaml")["valid_intervals"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of {valid_intervals}.")

        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")

        if start > end:
            raise ValueError("Start date should be before end date.")

        if interval == "1m" and start < datetime.now() - timedelta(days=30):
            raise ValueError(
                "For 1 min interval, the start date must be within 30 days from the current date."
            )

        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.data = self.get_data()

        if self.data is None or self.data.empty:
            raise ValueError("No data was found for the given parameters.")

        def get_data(self):
            pass
