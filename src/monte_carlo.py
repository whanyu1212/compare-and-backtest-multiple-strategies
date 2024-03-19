import numpy as np
from portfolio_stats import PortfolioStatistics


def monte_carlo_optimize_portfolio(price_df, symbols, num_iter=5000):
    np.random.seed(42)
    sharpe_ratio_list = []
    portfolio_ret_list = []
    portfolio_vol_list = []
    w_list = []

    max_sharpe = 0
    max_sharpe_vol = None
    max_sharpe_ret = None
    max_sharpe_w = None

    for i in range(num_iter):
        weights = np.random.random(len(symbols))
        weights /= np.sum(weights)
        portfolio = PortfolioStatistics(price_df, weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio.get_stats()

        if sharpe_ratio > max_sharpe:
            max_sharpe = sharpe_ratio
            max_sharpe_vol = portfolio_volatility
            max_sharpe_ret = portfolio_return
            max_sharpe_w = weights

        portfolio_vol_list.append(portfolio_volatility)
        portfolio_ret_list.append(portfolio_return)
        w_list.append(weights)
        sharpe_ratio_list.append(sharpe_ratio)

    return (
        max_sharpe,
        max_sharpe_vol,
        max_sharpe_ret,
        max_sharpe_w,
        portfolio_vol_list,
        portfolio_ret_list,
        w_list,
        sharpe_ratio_list,
    )
