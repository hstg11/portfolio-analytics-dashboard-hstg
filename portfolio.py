import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -------- Data preparation --------

def download_prices(tickers_list, start_date, end_date):
    """Download price data and return Close/Adj Close (wide) DataFrame."""
    raw = yf.download(
        tickers_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    data = raw["Close"]
    if "Adj Close" in raw.columns:
        data = raw["Adj Close"]
    return data.dropna()

def compute_returns(data: pd.DataFrame):
    """Compute daily returns from price data."""
    returns = data.pct_change().dropna()
    return returns

def portfolio_series(returns: pd.DataFrame, weights: np.ndarray):
    """Compute portfolio daily return series and cumulative index."""
    if returns.shape[1] != len(weights):
        weights = np.repeat(1 / returns.shape[1], returns.shape[1])
    port_ret = returns.dot(weights)
    cumulative = (1 + port_ret).cumprod()
    return port_ret, cumulative, weights

# -------- Risk metrics --------

def risk_metrics(portfolio_return: pd.Series, cumulative: pd.Series, risk_free_rate: float):
    """Compute annualized mu/sigma, Sharpe, max drawdown, VaR(95%)."""
    trading_days = 252
    mu_annual = portfolio_return.mean() * trading_days
    sigma_annual = portfolio_return.std() * np.sqrt(trading_days)
    sharpe = (mu_annual - risk_free_rate) / sigma_annual if sigma_annual != 0 else 0.0
    volatility = sigma_annual * 100

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    var_95 = portfolio_return.quantile(0.05) * 100

    return {
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
        "sharpe": sharpe,
        "volatility_pct": volatility,
        "max_drawdown_pct": max_drawdown,
        "var_95_pct": var_95,
        "trading_days": trading_days,
    }

def rolling_risk(portfolio_return: pd.Series, window: int, risk_free_rate: float):
    """Compute rolling annualized volatility and Sharpe."""
    trading_days = 252
    rolling_std = portfolio_return.rolling(window).std() * np.sqrt(trading_days)
    rolling_vol_pct = rolling_std * 100
    rolling_mean = portfolio_return.rolling(window).mean() * trading_days
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    return rolling_vol_pct, rolling_sharpe

# -------- Benchmark comparison --------

def benchmark_series(symbol: str, start_date, end_date):
    """Download benchmark close, returns, cumulative index."""
    benchmark = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]
    bench_ret = benchmark.pct_change().dropna()
    bench_cum = (1 + bench_ret).cumprod()
    return bench_ret, bench_cum

# -------- Monte Carlo --------

@st.cache_data(show_spinner=False)
def run_simulation(mu: float, sigma: float, num_days: int, num_sims: int):
    """Monte Carlo simulation paths of portfolio value (start at 1)."""
    start_value = 1.0
    sim_results = np.zeros((num_days, num_sims))
    for s in range(num_sims):
        daily_random_returns = np.random.normal(mu, sigma, num_days)
        sim_path = start_value * np.cumprod(1 + daily_random_returns)
        sim_results[:, s] = sim_path
    return sim_results
