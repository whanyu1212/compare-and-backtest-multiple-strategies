import numpy as np


def calculate_annualized_sharpe_ratio(returns, risk_free_rate):
    risk_free_rate_daily = risk_free_rate / 252
    excess_returns = returns - risk_free_rate_daily
    mean_excess_return_annualized = np.mean(excess_returns) * 252
    std_dev_excess_return_annualized = np.std(excess_returns) * np.sqrt(252)
    annualized_sharpe_ratio = (
        mean_excess_return_annualized / std_dev_excess_return_annualized
    )
    return annualized_sharpe_ratio


def calculate_annualized_sortino_ratio(returns, risk_free_rate):
    daily_excess_returns = returns - risk_free_rate
    annualized_excess_return = daily_excess_returns.mean() * 252
    downside_returns = daily_excess_returns[daily_excess_returns < 0]
    annualized_sortino_ratio = annualized_excess_return / (
        downside_returns.std() * np.sqrt(252)
    )
    return annualized_sortino_ratio


def calculate_annualized_treynor_ratio(returns, benchmark_returns, risk_free_rate):
    daily_excess_returns = returns - risk_free_rate
    annualized_excess_return = daily_excess_returns.mean() * 252
    beta = np.cov(returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
    annualized_treynor_ratio = annualized_excess_return / beta
    return annualized_treynor_ratio


def calculate_annualized_information_ratio(returns, benchmark_returns):
    daily_excess_returns = returns - benchmark_returns
    annualized_excess_return = daily_excess_returns.mean() * 252
    annualized_information_ratio = annualized_excess_return / (
        daily_excess_returns.std() * np.sqrt(252)
    )
    return annualized_information_ratio


def calculate_annualized_calmar_ratio(daily_returns, max_drawdown):
    annualized_return = np.prod(1 + daily_returns) ** (252 / len(daily_returns)) - 1
    calmar_ratio = annualized_return / abs(max_drawdown)
    return calmar_ratio
