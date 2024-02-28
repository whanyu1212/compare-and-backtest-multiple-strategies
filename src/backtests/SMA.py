import math
import pandas as pd
from typing import Union
from loguru import logger


class SMAVectorBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: Union[int, float],
        in_position: bool = False,
        sma1: int = 5,
        sma2: int = 8,
        sma3: int = 13,
    ):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty DataFrame")
        if not "Close" in df.columns:
            raise ValueError("No Close column found")
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number")
        if not isinstance(in_position, bool):
            raise ValueError("in_position must be a boolean")
        if not isinstance(sma1, int) or sma1 <= 0:
            raise ValueError("sma1 must be a positive integer")
        if not isinstance(sma2, int) or sma2 <= 0:
            raise ValueError("sma2 must be a positive integer")
        if not isinstance(sma3, int) or sma3 <= 0:
            raise ValueError("sma3 must be a positive integer")
        if not sma1 < sma2 < sma3:
            raise ValueError("sma1, sma2, and sma3 must be in ascending order")

        self.df = df
        self.sma1 = sma1
        self.sma2 = sma2
        self.sma3 = sma3
        self.in_position = in_position
        self.initial_capital = initial_capital
        self.equity = self.initial_capital
        self.no_of_shares = 0

    def calculate_moving_averages(self, df):
        df["sma1"] = df["Close"].rolling(window=self.sma1).mean()
        df["sma2"] = df["Close"].rolling(window=self.sma2).mean()
        df["sma3"] = df["Close"].rolling(window=self.sma3).mean()
        return df

    def buy_signal(self, df, i):
        if (
            df["sma2"][i - 1] < df["sma3"][i - 1]
            and df["sma2"][i] > df["sma3"][i]
            and df["sma1"][i] > df["sma2"][i]
            and df["Close"][i] > df["sma1"][i]
            and self.in_position == False
        ):
            self.no_of_shares = math.floor(self.equity / df.Close[i])
            self.equity -= self.no_of_shares * df.Close[i]
            self.in_position = True
        logger.info(
            f"{self.no_of_shares} shares bought at {df.Close[i]} on {df.index[i]}"
        )

    def sell_signal(self, df, i):
        if (
            df["sma2"][i - 1] > df["sma3"][i - 1]
            and df["sma2"][i] < df["sma3"][i]
            and df["sma1"][i] < df["sma2"][i]
            and df["Close"][i] < df["sma1"][i]
            and self.in_position == True
        ):
            self.equity += self.no_of_shares * df.Close[i]
            self.in_position = False
        logger.info(
            f"{self.no_of_shares} shares sold at {df.Close[i]} on {df.index[i]}"
        )

    def close_position(self, df, i):
        if self.in_position == True:
            self.equity += self.no_of_shares * df.Close[i]
            self.in_position = False
        logger.info(
            f"Closing position at {df.Close[i]} on {df.index[i]}. Equity is {self.equity}"
        )

    def calculate_buy_and_hold_roi(self):
        return round(
            (self.df["Close"].iloc[-1] - self.df["Close"].iloc[0])
            / self.df["Close"].iloc[0]
            * 100,
            2,
        )

    def calculate_earning_roi(self):
        return round(
            (self.equity - self.initial_capital) / self.initial_capital * 100, 2
        )

    def backtesting_strategy(self):
        df = self.df.copy()
        df = self.calculate_moving_averages(df)
        for i in range(1, len(df)):
            self.buy_signal(df, i)
            self.sell_signal(df, i)
        self.close_position(df, i)
        roi = self.calculate_earning_roi()
        return roi
