# optimizer.py

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import streamlit as st

# ---- Core math helpers ----

def portfolio_return_opt(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility_opt(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe(weights, mean_returns, cov_matrix, risk_free):
    ret = portfolio_return_opt(weights, mean_returns)
    vol = portfolio_volatility_opt(weights, cov_matrix)
    
    if vol == 0 or vol < 1e-8:  # ✅ Handle near-zero volatility
        return 1e10  # Large penalty instead of 0
    
    return -((ret - risk_free) / vol)
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

# ============================================
# EFFICIENT FRONTIER FUNCTIONS
# ============================================

@st.cache_data(show_spinner=False, ttl=600)
def generate_random_portfolios(
    mean_returns,
    cov_matrix,
    risk_free,
    num_portfolios=2000,
    lower_limit=0.0
):
    """
    Generate random portfolios for efficient frontier visualization.
    
    Returns DataFrame with: return, volatility, sharpe, weights
    """
    num_assets = len(mean_returns)
    results = {
        'return': [],
        'volatility': [],
        'sharpe': [],
        'weights': []
    }
    
    for _ in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        
        # Apply lower limit constraint
        weights = np.maximum(weights, lower_limit / 100.0)
        
        # Normalize to sum to 1
        weights = weights / np.sum(weights)
        
        # Calculate portfolio metrics
        port_return = portfolio_return_opt(weights, mean_returns)
        port_vol = portfolio_volatility_opt(weights, cov_matrix)
        sharpe = (port_return - risk_free) / port_vol if port_vol != 0 else 0.0
        
        # Store results
        results['return'].append(port_return)
        results['volatility'].append(port_vol)
        results['sharpe'].append(sharpe)
        results['weights'].append(weights)
    
    return pd.DataFrame(results)

@st.cache_data(show_spinner=False, ttl=600)
def get_efficient_frontier_curve(
    mean_returns,
    cov_matrix,
    risk_free,
    lower_limit=0.0,
    num_points=50
):
    """
    Generate points along the efficient frontier curve.
    Uses true min/max returns for deterministic frontier.
    """
    
    # ✅ Call your existing get_min_volatility_portfolio function
    from optimizer import get_min_volatility_portfolio  # Import if needed
    
    opt_weights, ret, vol, sharpe = get_min_volatility_portfolio(
        mean_returns, cov_matrix, risk_free, lower_limit
    )
    
    # Use this as min return
    min_return = ret
    max_return = mean_returns.max()
    
    # Rest stays the same...
    target_returns = np.linspace(min_return, max_return, num_points)
    
    efficient_portfolios = []
    
    for target_ret in target_returns:
        try:
            result = minimize_volatility_for_target_return(
                mean_returns,
                cov_matrix,
                target_ret,
                lower_limit
            )
            
            if result.success:
                weights = result.x
                vol = portfolio_volatility_opt(weights, cov_matrix)
                ret = portfolio_return_opt(weights, mean_returns)
                sharpe = (ret - risk_free) / vol if vol != 0 else 0.0
                
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': weights
                })
        except:
            continue
    
    return pd.DataFrame(efficient_portfolios)

def minimize_volatility_for_target_return(
    mean_returns,
    cov_matrix,
    target_return,
    lower_limit=0.0
):
    """
    Minimize portfolio volatility while achieving a target return.
    This creates one point on the efficient frontier.
    """
    num_assets = len(mean_returns)
    init_weights = np.array(num_assets * [1.0 / num_assets])
    
    # Constraints:
    # 1. Weights sum to 1
    # 2. Portfolio return equals target return
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_return_opt(w, mean_returns) - target_return}
    )
    
    # Bounds: each weight between lower_limit and 100%
    bounds = tuple((lower_limit / 100.0, 1.0) for _ in range(num_assets))
    
    result = minimize(
        portfolio_volatility_opt,
        init_weights,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    return result


def find_optimal_portfolio_at_risk(
    efficient_df,
    target_risk
):
    """
    Find the best portfolio on the efficient frontier 
    that has volatility <= target_risk.
    
    Returns the portfolio with highest return within risk tolerance.
    """
    # Filter portfolios within risk tolerance
    within_tolerance = efficient_df[
        efficient_df['volatility'] * 100 <= target_risk
    ]
    
    if within_tolerance.empty:
        return None
    
    # Return the one with highest return
    best_idx = within_tolerance['return'].idxmax()
    return within_tolerance.loc[best_idx]
