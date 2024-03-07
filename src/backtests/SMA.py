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
        SMA1: int = 2,
        SMA2: int = 3,
        SMA3: int = 19,
    ):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty DataFrame")
        if not "Close" in df.columns:
            raise ValueError("No Close column found")
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number")
        if not isinstance(in_position, bool):
            raise ValueError("in_position must be a boolean")
        if not isinstance(SMA1, int) or SMA1 <= 0:
            raise ValueError("SMA1 must be a positive integer")
        if not isinstance(SMA2, int) or SMA2 <= 0:
            raise ValueError("SMA2 must be a positive integer")
        if not isinstance(SMA3, int) or SMA3 <= 0:
            raise ValueError("SMA3 must be a positive integer")
        if not SMA1 < SMA2 < SMA3:
            raise ValueError("SMA1, SMA2, and SMA3 must be in ascending order")

        self.df = df.drop(["Open", "High", "Low", "Volume"], axis=1)
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.SMA3 = SMA3
        self.in_position = in_position
        self.initial_capital = initial_capital
        self.balance = self.initial_capital
        self.no_of_shares = 0

    def calculate_moving_averages(self, df):
        df["SMA1"] = df["Close"].rolling(window=self.SMA1).mean()
        df["SMA2"] = df["Close"].rolling(window=self.SMA2).mean()
        df["SMA3"] = df["Close"].rolling(window=self.SMA3).mean()
        return df

    def calculate_daily_returns(self, df):
        df["Price Returns"] = df["Close"].pct_change()
        return df

    def create_columns(self, df):
        initial_values = {
            "In Position": (self.in_position, "bool"),
            "Balance": (self.balance, "float64"),
            "No of Shares": (self.no_of_shares, "float64"),
        }

        for column, (initial_value, dtype) in initial_values.items():
            df[column] = pd.Series(dtype=dtype)
            df.loc[0, column] = initial_value

        return df

    def buy_signal(self, df, i):
        if (
            df["SMA2"][i - 1] < df["SMA3"][i - 1]
            and df["SMA2"][i] > df["SMA3"][i]
            and df["SMA1"][i] > df["SMA2"][i]
            and df["Close"][i] > df["SMA1"][i]
            and self.in_position == False
        ):
            return True

    def sell_signal(self, df, i):
        if (
            df["SMA2"][i - 1] > df["SMA3"][i - 1]
            and df["SMA2"][i] < df["SMA3"][i]
            and df["SMA1"][i] < df["SMA2"][i]
            and df["Close"][i] < df["SMA1"][i]
            and self.in_position == True
        ):
            return True

    def backtest_strategy(self, df):
        for i in range(1, len(df)):
            if self.buy_signal(df, i):
                self.no_of_shares = math.floor(self.balance / df.loc[i, "Close"])
                self.balance -= self.no_of_shares * df.loc[i, "Close"]
                self.in_position = True
            elif self.sell_signal(df, i):
                self.balance += self.no_of_shares * df.loc[i, "Close"]
                self.no_of_shares = 0
                self.in_position = False

            df.loc[i, "No of Shares"] = self.no_of_shares
            df.loc[i, "Balance"] = self.balance
            df.loc[i, "In Position"] = self.in_position

        if self.in_position:
            self.balance += self.no_of_shares * df.loc[df.index[-1], "Close"]
            self.no_of_shares = 0
            self.in_position = False
            df.loc[df.index[-1], "No of Shares"] = self.no_of_shares
            df.loc[df.index[-1], "Balance"] = self.balance
            df.loc[df.index[-1], "In Position"] = self.in_position

        return df

    def create_signal_column(self, df):
        df["In Position Shift"] = df["In Position"].shift(1)
        df.loc[
            (df["In Position"] == True) & (df["In Position Shift"] == False), "Signal"
        ] = "Buy"
        df.loc[
            (df["In Position"] == False) & (df["In Position Shift"] == True), "Signal"
        ] = "Sell"
        df["Signal"].fillna("Hold", inplace=True)
        return df

    def calculate_strategy_capital_change(self, df):
        df["Total Strategy Capital"] = df["Balance"] + df["No of Shares"] * df["Close"]
        return df

    def calculate_buy_and_hold_capital_change(self, df):
        left_over = (
            self.initial_capital
            - math.floor(self.initial_capital / df.loc[0, "Close"]) * df.loc[0, "Close"]
        )
        df["Total Buy and Hold Capital"] = (
            self.initial_capital * (1 + df["Price Returns"]).cumprod() + left_over
        )
        return df

    def calculate_realized_pnl(self, df):
        df_subset = df.copy().query("Signal!='Hold'")
        df_subset["Signal Shift"] = df_subset["Signal"].shift(1)
        df_subset.loc[
            (df_subset["Signal"] == "Sell") & (df_subset["Signal Shift"] == "Buy"),
            "Realized PnL",
        ] = df_subset["Total Strategy Capital"] - df_subset[
            "Total Strategy Capital"
        ].shift(
            1
        )
        df = df.merge(df_subset[["Date", "Realized PnL"]], on="Date", how="left")
        return df

    def calculate_unrealized_pnl(self, df):
        df_subset_2 = df.copy().query("Signal!='Sell'")

        reference = 0
        for index, row in df_subset_2.iterrows():
            if row["Signal"] == "Buy":
                reference = row["Total Strategy Capital"]
            elif row["Signal"] == "Hold" and row["No of Shares"] > 0:
                df_subset_2.loc[index, "Unrealized PnL"] = (
                    row["Total Strategy Capital"] - reference
                )

        df = df.merge(df_subset_2[["Date", "Unrealized PnL"]], on="Date", how="left")
        return df

    def reorganize_columns(self, df):
        columns = [
            "Date",
            "Close",
            "SMA1",
            "SMA2",
            "SMA3",
            "Price Returns",
            "Signal",
            "In Position",
            "No of Shares",
            "Balance",
            "Realized PnL",
            "Unrealized PnL",
            "Total Strategy Capital",
            "Total Buy and Hold Capital",
        ]
        df = df[columns]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df

    def backtesting_flow(self):
        df = self.df.copy().reset_index()
        df = self.calculate_moving_averages(df)
        df = self.calculate_daily_returns(df)
        df = self.create_columns(df)
        df = self.backtest_strategy(df)
        df = self.create_signal_column(df)
        df = self.calculate_strategy_capital_change(df)
        df = self.calculate_buy_and_hold_capital_change(df)
        df = self.calculate_realized_pnl(df)
        df = self.calculate_unrealized_pnl(df)
        df = self.reorganize_columns(df)
        return df
