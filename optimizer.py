import numpy as np
from scipy.optimize import minimize

def portfolio_return_opt(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility_opt(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free):
    ret = portfolio_return_opt(weights, mean_returns)
    vol = portfolio_volatility_opt(weights, cov_matrix)
    return -((ret - risk_free) / vol if vol != 0 else 0.0)

def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free):
    num_assets = len(mean_returns)
    init_weights = np.array(num_assets * [1.0 / num_assets])
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(
        neg_sharpe,
        init_weights,
        args=(mean_returns, cov_matrix, risk_free),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result

def get_max_sharpe_portfolio(mean_returns, cov_matrix, risk_free):
    result = max_sharpe_portfolio(mean_returns, cov_matrix, risk_free)
    opt_weights = result.x
    ret = portfolio_return_opt(opt_weights, mean_returns)
    vol = portfolio_volatility_opt(opt_weights, cov_matrix)
    sharpe = (ret - risk_free) / vol if vol != 0 else 0.0
    return opt_weights, ret, vol, sharpe

def portfolio_stats(weights, mean_returns, cov_matrix, risk_free):
    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free) / port_vol if port_vol != 0 else 0.0
    return port_return, port_vol, sharpe
