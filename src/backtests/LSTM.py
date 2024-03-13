import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import *
from typing import Union
from util.utility_functions import parse_cfg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LSTMVectorBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: Union[int, float],
        in_position: bool = False,
        time_step: int = 5,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df
        self.initial_capital = initial_capital
        self.in_position = in_position
        self.balance = self.initial_capital
        self.time_step = time_step
        self.no_of_shares = 0
        self.base_columns = parse_cfg("./config/parameters.yaml")["base_columns"]

    def calculate_daily_returns(self, df):
        df["Market Returns"] = df["Close"].pct_change()
        return df

    def scaling_df(self, df):
        self.scaler = MinMaxScaler()
        df_scaled = self.scaler.fit_transform(df["Close"].values.reshape(-1, 1))
        return df_scaled

    def create_dataset(self, dataset):
        X_data, y_data = [], []
        for i in range(len(dataset) - self.time_step - 1):
            X_data.append(dataset[i : (i + self.time_step), 0])
            y_data.append(dataset[i + self.time_step, 0])
        return np.array(X_data), np.array(y_data)

    def reshape_x_input(self, X_data):
        return np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

    def create_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer="adam", loss="mean_absolute_error")

        return model

    def fit_model(self, X_train, y_train, model):
        model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

    def add_predicted_columns(self, df, X_test, model):
        y_pred = model.predict(X_test)
        y_pred = self.scaler.inverse_transform(y_pred)
        df["Predicted Close"] = [np.nan] * (
            self.time_step + 1
        ) + y_pred.flatten().tolist()
        df["Predicted Returns"] = df["Predicted Close"].pct_change()

        return df.reset_index()

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
        df = self.calculate_daily_returns(self.df)
        df_scaled = self.scaling_df(df)
        X_data, y_data = self.create_dataset(df_scaled)
        X_data_reshaped = self.reshape_x_input(X_data)
        model = self.create_model(input_shape=(X_data_reshaped.shape[1], 1))
        self.fit_model(X_data_reshaped, y_data, model)
        df = self.add_predicted_columns(df, X_data_reshaped, model)
        df = self.create_columns(df)
        return df

    def buy_signal(self, df, i):
        if df["Predicted Returns"][i] > 0.001 and self.in_position == False:
            return True

    def sell_signal(self, df, i):
        if df["Predicted Returns"][i] < -0.015 and self.in_position == True:
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

    def reorganize_columns(
        self, df, additional_columns=["Predicted Close", "Predicted Returns"]
    ):
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
