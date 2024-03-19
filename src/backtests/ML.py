import pandas as pd
import numpy as np
import shap
import math
from typing import Union
from lightgbm import LGBMClassifier
from util.utility_functions import parse_cfg


class MLClassifierVectorBacktester:
    def __init__(
        self,
        df: pd.DataFrame,
        initial_capital: Union[int, float],
        in_position: bool = False,
        shift_days: int = 10,
    ):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("df must be a non-empty DataFrame")
        if not "Close" in df.columns:
            raise ValueError("No Close column found")
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number")
        if not isinstance(in_position, bool):
            raise ValueError("in_position must be a boolean")
        if not isinstance(shift_days, int) or shift_days <= 0:
            raise ValueError("shift_days must be a positive integer")

        self.df = df
        self.in_position = in_position
        self.initial_capital = initial_capital
        self.balance = self.initial_capital
        self.no_of_shares = 0
        self.shift_days = shift_days
        self.base_columns = parse_cfg("./config/parameters.yaml")["base_columns"]

    def calculate_daily_returns(self, df):
        df["Market Returns"] = df["Close"].pct_change()
        return df

    def calculate_technical_indicators(self, data):
        data["SMA50"] = data["Close"].rolling(window=50).mean()
        data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()

        # RSI
        delta = data["Close"].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data["Close"].ewm(span=12, adjust=False).mean()
        exp2 = data["Close"].ewm(span=26, adjust=False).mean()
        data["MACD"] = exp1 - exp2

        # ATR
        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Close"].shift()).abs()
        low_close = (data["Low"] - data["Close"].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data["ATR"] = true_range.rolling(window=14).mean()

        # Bollinger Bands
        data["Middle_BB"] = data["Close"].rolling(window=20).mean()
        data["Upper_BB"] = (
            data["Middle_BB"] + 2 * data["Close"].rolling(window=20).std()
        )
        data["Lower_BB"] = (
            data["Middle_BB"] - 2 * data["Close"].rolling(window=20).std()
        )

        # OBV
        data["OBV"] = (
            (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()
        )

        # VWAP (assuming intraday data is available)
        data["VWAP"] = (
            data["Volume"] * (data["High"] + data["Low"] + data["Close"]) / 3
        ).cumsum() / data["Volume"].cumsum()

        return data

    def create_target(self, data):
        data["Future_Close"] = data["Close"].shift(-self.shift_days)
        data["Target"] = (data["Future_Close"] > data["Close"].shift(-1)).astype(int)
        return data

    def predictor_response_split(self, data):
        X = data.drop(
            ["Symbol", "Market Returns", "Date", "Future_Close", "Target"], axis=1
        )
        y = data["Target"]
        return X, y

    def fit_classifier(self, X, y):
        self.model = LGBMClassifier()
        self.model.fit(X, y)

    def calculate_shap_values(self, X):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        return shap_values

    def create_shap_df(self, X, shap_values):
        shap_df = pd.DataFrame(
            list(zip(X.columns, np.abs(shap_values).mean(0))),
            columns=["col_name", "shap_values"],
        )
        shap_df.sort_values(by="shap_values", ascending=False, inplace=True)
        return shap_df

    def get_top_n_features(self, shap_df, n=5):
        top_features = shap_df["col_name"][:n].tolist()
        return top_features

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

    def prepare_data(self, data):
        data = self.calculate_daily_returns(data)
        data = self.calculate_technical_indicators(data)
        data = self.create_target(data)
        X, y = self.predictor_response_split(data)
        self.fit_classifier(X, y)
        shap_values = self.calculate_shap_values(X)
        shap_df = self.create_shap_df(X, shap_values)
        top_features = self.get_top_n_features(shap_df)
        data = self.create_columns(data)
        return data, top_features, shap_df

    def backtest_strategy(self, df, selected_features):
        self.model = LGBMClassifier()
        self.model.fit(df[selected_features], df["Target"])
        for i, row in df.iterrows():
            prob_negative, prob_positive = self.model.predict_proba(
                [row[selected_features].values]
            )[0]
            if prob_positive > 0.9 and self.in_position == False:
                self.no_of_shares = math.floor(self.balance / df.loc[i, "Close"])
                self.balance -= self.no_of_shares * df.loc[i, "Close"]
                self.in_position = True

            elif prob_negative > (1 - 0.1) and self.in_position == True:
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

    def reorganize_columns(self, df, additional_columns=[]):
        target_columns = self.base_columns + additional_columns
        df = df[target_columns]
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        return df

    def backtesting_flow(self):
        df = self.df.copy().reset_index()
        df, selected_features, shap_df = self.prepare_data(df)
        df = self.backtest_strategy(df, selected_features)
        df = self.post_process(df)
        df = self.reorganize_columns(df)
        return df, shap_df
