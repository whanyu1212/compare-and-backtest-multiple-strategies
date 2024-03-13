import pandas as pd
import numpy as np


class PortfolioStatistics:
    def __init__(self, price_df, weights):
        self.price_df = price_df
        self.weights = weights
        self.daily_return = self.calculate_daily_return()
        self.portfolio_return = self.calculate_portfolio_return()
        self.portfolio_volatility = self.calculate_portfolio_volatility()
        self.portfolio_sharpe_ratio = self.calculate_sharpe_ratio()

    def calculate_daily_return(self):
        daily_return = self.price_df.pct_change()
        return daily_return

    def calculate_portfolio_return(self):
        portfolio_return = self.daily_return.mean() * 252
        return portfolio_return

    def calculate_portfolio_volatility(self):
        portfolio_volatility = self.daily_return.cov() * 252
        return portfolio_volatility

    def calculate_sharpe_ratio(self):
        sharpe_ratio = self.portfolio_return / self.portfolio_volatility
        return sharpe_ratio

    def get_stats(self):
        weights = np.array(self.weights)
        portfolio_return = np.sum(self.daily_return.mean() * weights) * 252
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.daily_return.cov() * 252, weights))
        )
        sharpe_ratio = portfolio_return / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio
