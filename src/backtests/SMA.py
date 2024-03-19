import math
import pandas as pd
from typing import Union
from util.utility_functions import parse_cfg


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
        self.base_columns = parse_cfg("./config/parameters.yaml")["base_columns"]

    def calculate_moving_averages(self, df):
        df["SMA1"] = df["Close"].rolling(window=self.SMA1).mean()
        df["SMA2"] = df["Close"].rolling(window=self.SMA2).mean()
        df["SMA3"] = df["Close"].rolling(window=self.SMA3).mean()
        return df

    def calculate_daily_returns(self, df):
        df["Market Returns"] = df["Close"].pct_change()
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

    def prepare_data(self, df):
        df = self.calculate_moving_averages(df)
        df = self.calculate_daily_returns(df)
        df = self.create_columns(df)
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
        """Generate trading signals based on position changes."""
        df["Signal"] = "Hold"  # Default to 'Hold'
        df.loc[
            (df["In Position"] == True) & (df["In Position"].shift(1) == False),
            "Signal",
        ] = "Buy"
        df.loc[
            (df["In Position"] == False) & (df["In Position"].shift(1) == True),
            "Signal",
        ] = "Sell"
        return df

    def calculate_total_strategy_portfolio_value(self, df):
        """Calculate the total portfolio value for the strategy, including cash and the value of held shares."""
        df["Total Strategy Portfolio Value"] = (
            df["Balance"] + df["No of Shares"] * df["Close"]
        )
        return df

    def calculate_buy_and_hold_portfolio_value(self, df):
        """Calculate the total portfolio value for a buy-and-hold strategy."""
        shares_bought = math.floor(self.initial_capital / df.loc[0, "Close"])
        left_over = self.initial_capital - (shares_bought * df.loc[0, "Close"])
        df["Buy and Hold Portfolio Value"] = (shares_bought * df["Close"]) + left_over
        df["Buy and Hold Portfolio Value"].fillna(self.initial_capital, inplace=True)
        return df

    def calculate_realized_pnl(self, df):
        sell_signals = df["Signal"] == "Sell"
        buy_signals = df["Signal"] == "Buy"
        sell_following_buy = sell_signals & buy_signals.shift(1).fillna(False)
        df.loc[sell_following_buy, "Realized PnL"] = df[
            "Total Strategy Portfolio Value"
        ] - df["Total Strategy Portfolio Value"].shift(1)

        df["Realized PnL"].fillna(0, inplace=True)
        return df

    def calculate_unrealized_pnl(self, df):
        df["Unrealized PnL"] = 0

        # Track the buy price for the latest open position
        buy_price_per_share = None

        for i, row in df.iterrows():
            if row["Signal"] == "Buy":
                buy_price_per_share = (
                    row["Close"] / self.no_of_shares if self.no_of_shares > 0 else None
                )
            elif row["Signal"] == "Sell":
                buy_price_per_share = None

            if row["In Position"] and buy_price_per_share is not None:
                current_value_of_shares = row["No of Shares"] * row["Close"]
                cost_of_held_shares = row["No of Shares"] * buy_price_per_share
                df.at[i, "Unrealized PnL"] = (
                    current_value_of_shares - cost_of_held_shares
                )

        return df

    def calculate_drawdowns(self, df, capital_column):
        cumulative_max = df[capital_column].cummax()
        drawdown = (df[capital_column] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return drawdown, max_drawdown

    def post_process(self, df):
        df = self.create_signal_column(df)
        df = self.calculate_total_strategy_portfolio_value(df)
        df = self.calculate_buy_and_hold_portfolio_value(df)
        df = self.calculate_realized_pnl(df)
        df = self.calculate_unrealized_pnl(df)
        df["Strategy Drawdown"], df["Strategy Max Drawdown"] = self.calculate_drawdowns(
            df, "Total Strategy Portfolio Value"
        )
        df["Buy and Hold Drawdown"], df["Buy and Hold Max Drawdown"] = (
            self.calculate_drawdowns(df, "Buy and Hold Portfolio Value")
        )
        return df

    def reorganize_columns(self, df, additional_columns=["SMA1", "SMA2", "SMA3"]):
        target_columns = self.base_columns + additional_columns
        df = df[target_columns]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df

    def backtesting_flow(self):
        df = self.df.copy().reset_index()
        df = self.prepare_data(df)
        df = self.backtest_strategy(df)
        df = self.post_process(df)
        df = self.reorganize_columns(df)
        return df
