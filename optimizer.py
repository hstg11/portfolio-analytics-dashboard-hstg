# optimizer.py

import numpy as np
from scipy.optimize import minimize

# ---- Core math helpers ----

def portfolio_return_opt(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility_opt(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free):
    ret = portfolio_return_opt(weights, mean_returns)
    vol = portfolio_volatility_opt(weights, cov_matrix)
    return -((ret - risk_free) / vol if vol != 0 else 0.0)

# ---- Max Sharpe optimization ----

def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free, lower_limit=0.0):
    """
    SLSQP maximize Sharpe through minimizing negative Sharpe.
    lower_limit in percent (e.g., 5.0 for 5%), same as app's slider behavior.
    """
    num_assets = len(mean_returns)
    init_weights = np.array(num_assets * [1.0 / num_assets])
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((lower_limit / 100.0, 1.0) for _ in range(num_assets))

    result = minimize(
        neg_sharpe,
        init_weights,
        args=(mean_returns, cov_matrix, risk_free),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result

def get_max_sharpe_portfolio(mean_returns, cov_matrix, risk_free, lower_limit=0.0):
    result = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free, lower_limit)
    opt_weights = result.x
    ret = portfolio_return_opt(opt_weights, mean_returns)
    vol = portfolio_volatility_opt(opt_weights, cov_matrix)
    sharpe = (ret - risk_free) / vol if vol != 0 else 0.0
    return opt_weights, ret, vol, sharpe

# ---- Min Volatility optimization ----

def min_volatility_portfolio(mean_returns, cov_matrix, lower_limit=0.0):
    """
    SLSQP minimize portfolio volatility with sum(weights)=1 and per-asset lower bound.
    lower_limit in percent (e.g., 5.0 for 5%), same as app's slider behavior.
    """
    num_assets = len(mean_returns)
    init_weights = np.array(num_assets * [1.0 / num_assets])
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((lower_limit / 100.0, 1.0) for _ in range(num_assets))

    result = minimize(
        portfolio_volatility_opt,
        init_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result

def get_min_volatility_portfolio(mean_returns, cov_matrix, risk_free, lower_limit=0.0):
    result = min_volatility_portfolio(mean_returns, cov_matrix, lower_limit)
    opt_weights = result.x
    ret = portfolio_return_opt(opt_weights, mean_returns)
    vol = portfolio_volatility_opt(opt_weights, cov_matrix)
    sharpe = (ret - risk_free) / vol if vol != 0 else 0.0
    return opt_weights, ret, vol, sharpe

# ---- Stats helper (unchanged from app) ----

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free) / port_vol if port_vol != 0 else 0.0
    return port_return, port_vol, sharpe
