import math
import pandas as pd
import numpy as np
from typing import Union
from util.utility_functions import parse_cfg


class SuperTrendVectorBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: Union[int, float],
        in_position: bool = False,
        period=8,
        multiplier=2,
    ):
        self.df = df
        self.initial_capital = initial_capital
        self.in_position = in_position
        self.period = period
        self.multiplier = multiplier
        self.balance = self.initial_capital
        self.no_of_shares = 0
        self.base_columns = parse_cfg("./config/parameters.yaml")["base_columns"]

    def calculate_atr(self, df):
        df["High-Low"] = df["High"] - df["Low"]
        df["High-PrevClose"] = abs(df["High"] - df["Close"].shift(1))
        df["Low-PrevClose"] = abs(df["Low"] - df["Close"].shift(1))
        df["TR"] = df[["High-Low", "High-PrevClose", "Low-PrevClose"]].max(axis=1)
        df["ATR"] = df["TR"].rolling(self.period).mean()
        return df

    def calculate_supertrend(self, df):
        self.calculate_atr(df)
        hl2 = (df["High"] + df["Low"]) / 2
        df["Final Upperband"] = hl2 + (self.multiplier * df["ATR"])
        df["Final Lowerband"] = hl2 - (self.multiplier * df["ATR"])
        df["Supertrend"] = np.nan

        for current in range(1, len(df)):
            previous = current - 1
            if df["Close"].iloc[current] > df["Final Upperband"].iloc[previous]:
                df.loc[current, "Supertrend"] = df["Final Lowerband"].iloc[current]
            elif df["Close"].iloc[current] < df["Final Lowerband"].iloc[previous]:
                df.loc[current, "Supertrend"] = df["Final Upperband"].iloc[current]
            else:
                df.loc[current, "Supertrend"] = df["Supertrend"].iloc[previous]
                if (
                    df["Supertrend"].iloc[current]
                    == df["Final Upperband"].iloc[previous]
                    and df["Close"].iloc[current] <= df["Final Upperband"].iloc[current]
                ):
                    df.loc[current, "Supertrend"] = df["Final Lowerband"].iloc[current]
                elif (
                    df["Supertrend"].iloc[current]
                    == df["Final Lowerband"].iloc[previous]
                    and df["Close"].iloc[current] >= df["Final Lowerband"].iloc[current]
                ):
                    df.loc[current, "Supertrend"] = df["Final Upperband"].iloc[current]
        return df

    def calculate_daily_returns(self, df):
        df["Market Returns"] = df["Close"].pct_change()
        return df

    def create_additional_columns(self, df):
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
        df = self.calculate_supertrend(df)
        df = self.calculate_daily_returns(df)
        df = self.create_additional_columns(df)
        return df

    def buy_signal(self, df, i):
        if df["Close"][i] > df["Supertrend"][i] and not self.in_position:
            return True

    def sell_signal(self, df, i):
        if df["Supertrend"][i] > df["Close"][i] and self.in_position:
            return True

    def backtest_strategy(self, df):
        for i in range(1, len(df)):
            close_price = df.loc[i, "Close"]
            if self.buy_signal(df, i):
                self.no_of_shares = math.floor(self.balance / close_price)
                self.balance -= self.no_of_shares * close_price
                self.in_position = True
            elif self.sell_signal(df, i):
                self.balance += self.no_of_shares * close_price
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
        """Calculate realized profit and loss."""
        # Identify rows with sell signals that follow buy signals directly
        sell_signals = df["Signal"] == "Sell"
        buy_signals = df["Signal"] == "Buy"
        sell_following_buy = sell_signals & buy_signals.shift(1).fillna(False)

        # Calculate P&L only on rows with a sell signal that follows a buy signal
        df.loc[sell_following_buy, "Realized PnL"] = df[
            "Total Strategy Portfolio Value"
        ] - df["Total Strategy Portfolio Value"].shift(1)

        # Fill missing values with 0 or appropriate method
        df["Realized PnL"].fillna(0, inplace=True)
        return df

    def calculate_unrealized_pnl(self, df):
        """Calculate unrealized profit and loss for open positions."""
        # Initialize Unrealized PnL column
        df["Unrealized PnL"] = 0

        # Track the buy price for the latest open position
        buy_price_per_share = None

        for i, row in df.iterrows():
            if row["Signal"] == "Buy":
                # Update buy price per share when a new position is opened
                buy_price_per_share = (
                    row["Close"] / self.no_of_shares if self.no_of_shares > 0 else None
                )
            elif row["Signal"] == "Sell":
                # Reset buy price per share when the position is closed
                buy_price_per_share = None

            if row["In Position"] and buy_price_per_share is not None:
                # Calculate current value of held shares
                current_value_of_shares = row["No of Shares"] * row["Close"]
                # Calculate cost of held shares
                cost_of_held_shares = row["No of Shares"] * buy_price_per_share
                # Update Unrealized PnL
                df.at[i, "Unrealized PnL"] = (
                    current_value_of_shares - cost_of_held_shares
                )

        return df

    def calculate_drawdowns(self, df, capital_column):
        """Calculates drawdown and max drawdown for a given capital column."""
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

    def reorganize_columns(self, df, additional_columns=["Supertrend"]):
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
