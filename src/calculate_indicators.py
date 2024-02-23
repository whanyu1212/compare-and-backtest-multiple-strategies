import pandas as pd
import numpy as np
from typing import List


class FinancialIndicators:
    def __init__(self, data, windows, lags):
        self.data = data
        self.windows = windows
        self.lags = lags

    def calculate_price_change(self) -> pd.DataFrame:
        """Calculate the magnitude of the price change
        compare to the previous day.

        Returns:
            pd.DataFrame: _description_
        """
        self.data["Price Change"] = self.data["Close"] - self.data["Close"].shift(1)
        return self.data

    def calculate_price_moving_average(self) -> pd.DataFrame:
        """Calculate simple moving average across different
        window sizes

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        for window in self.windows:
            self.data[f"SMA_{window}"] = (
                self.data["Close"].rolling(window=window).mean()
            )
        return self.data

    def calculate_returns(self):
        self.data["Returns"] = self.data["Close"].pct_change()
        return self.data

    def calculate_returns_moving_average(self) -> pd.DataFrame:
        """Calculate the moving average of the returns.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        for window in self.windows:
            self.data[f"Returns_{window}"] = (
                self.data["Returns"].rolling(window=window).mean()
            )
        return self.data

    def calculate_lagged_returns(self) -> pd.DataFrame:
        """Calculate the lagged returns.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        for lag in self.lags:
            self.data[f"Lagged_Returns_{lag}"] = self.data["Returns"].shift(lag)
        return self.data

    def calculate_RSI(self) -> pd.DataFrame:
        """Calculate the RSI given the window period.

        Returns:
            pd.DataFrame: dataframe with the added column
        """
        for window in self.windows:
            delta = self.data["Close"].diff()
            up, down = delta.copy(), delta.copy()
            up[up < 0] = 0
            down[down > 0] = 0

            average_gain = up.rolling(window).mean()
            average_loss = abs(down.rolling(window).mean())

            rs = average_gain / average_loss
            self.data[f"RSI_{window}"] = 100 - (100 / (1 + rs))
        return self.data

    def calculate_EMA(self) -> pd.DataFrame:
        """Calculate the exponential moving average across
        different window sizes.

        Returns:
            pd.DataFrame: dataframe with the added column
        """
        for window in self.windows:
            self.data[f"EMA_{window}"] = (
                self.data["Close"].ewm(span=window, adjust=False).mean()
            )
        return self.data

    def calculate_ATR(self) -> pd.DataFrame:
        """Calculate the average true range given the window period.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        self.data = self.data.assign(
            HLC=(self.data["High"] - self.data["Low"]),
            HL=(self.data["High"] - self.data["Close"].shift(1)).abs(),
            LC=(self.data["Low"] - self.data["Close"].shift(1)).abs(),
        )
        self.data["TR"] = self.data[["HLC", "HL", "LC"]].max(axis=1)
        for window in self.windows:
            self.data[f"ATR_{window}"] = self.data["TR"].rolling(window).mean()
        return self.data

    def calculate_MACD(self) -> pd.DataFrame:
        """Calculate the moving average convergence divergence.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        self.data["26 EMA"] = self.data["Close"].ewm(span=26, adjust=False).mean()
        self.data["12 EMA"] = self.data["Close"].ewm(span=12, adjust=False).mean()
        self.data["MACD"] = self.data["12 EMA"] - self.data["26 EMA"]
        self.data["Signal Line"] = self.data["MACD"].ewm(span=9, adjust=False).mean()
        return self.data

    def calculate_bollinger_bands(self, num_std: int = 2) -> pd.DataFrame:
        for window in self.windows:
            self.data[f"BB_Lower_{window}"] = (
                self.data["Close"].rolling(window=window).mean()
                - num_std * self.data["Close"].rolling(window=window).std()
            )
            self.data[f"BB_Upper_{window}"] = (
                self.data["Close"].rolling(window=window).mean()
                + num_std * self.data["Close"].rolling(window=window).std()
            )
        return self.data

    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all the indicators.

        Returns:
            pd.DataFrame: dataframe with the added columns
        """
        self.calculate_price_change()
        self.calculate_price_moving_average()
        self.calculate_returns()
        self.calculate_returns_moving_average()
        self.calculate_lagged_returns()
        self.calculate_RSI()
        self.calculate_EMA()
        self.calculate_ATR()
        self.calculate_MACD()
        self.calculate_bollinger_bands()
        return self.data
