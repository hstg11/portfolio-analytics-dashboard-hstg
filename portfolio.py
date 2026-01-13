#portfolio.py
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

def risk_metrics(
    portfolio_return: pd.Series,
    cumulative: pd.Series,
    risk_free_rate: float
):
    """
    Compute portfolio risk metrics:
    - Annualized return & volatility
    - Sharpe ratio
    - Sortino ratio
    - Max drawdown
    - VaR (95%, 99%)
    - CVaR / TVaR (95%, 99%)
    """

    trading_days = 252
    mu_annual = portfolio_return.mean() * trading_days
    sigma_annual = portfolio_return.std() * np.sqrt(trading_days)
    volatility_pct = sigma_annual * 100

    # =========================
    # Sharpe Ratio
    # =========================
    sharpe = (
        (mu_annual - risk_free_rate) / sigma_annual
        if sigma_annual != 0 else np.nan
    )

    # =========================
    # Sortino Ratio (downside risk)
    # =========================
    daily_rf = risk_free_rate / trading_days
    downside_returns = portfolio_return[portfolio_return < daily_rf] - daily_rf

    downside_dev = (
        np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(trading_days)
        if len(downside_returns) > 0 else np.nan
    )

    sortino = (
        (mu_annual - risk_free_rate) / downside_dev
        if downside_dev and downside_dev != 0 else np.nan
    )

    # =========================
    # Drawdown
    # =========================
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown_pct = drawdown.min() * 100

    # =========================
    # VaR (Historical)
    # =========================
    var_95 = portfolio_return.quantile(0.05)
    var_99 = portfolio_return.quantile(0.01)

    # =========================
    # CVaR / TVaR
    # =========================
    cvar_95 = portfolio_return[portfolio_return <= var_95].mean()
    cvar_99 = portfolio_return[portfolio_return <= var_99].mean()

    return {
        # Core return & risk
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
        "volatility_pct": volatility_pct,

        # Ratios
        "sharpe": sharpe,
        "sortino": sortino,

        # Drawdown
        "max_drawdown_pct": max_drawdown_pct,

        # Tail risk
        "var_95_pct": var_95 * 100,
        "var_99_pct": var_99 * 100,
        "cvar_95_pct": cvar_95 * 100,
        "cvar_99_pct": cvar_99 * 100,

        # Meta
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
    # ✅ Add 1 day buffer like portfolio
    adjusted_start = start_date - pd.Timedelta(days=1)
    
    benchmark = yf.download(
        symbol,
        start=adjusted_start,  # Match portfolio behavior
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]
    
    bench_ret = benchmark.pct_change().dropna()
    bench_cum = (1 + bench_ret).cumprod()
    return bench_ret, bench_cum

def portfolio_beta(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Compute Portfolio Beta relative to a benchmark.

    Beta = Covariance(portfolio, market) / Variance(market)
    """

    # Align dates to avoid mismatch
    df = pd.concat(
        [portfolio_returns, benchmark_returns],
        axis=1,
        join="inner"
    ).dropna()

    df.columns = ["portfolio", "benchmark"]

    # Safety check
    if df.shape[0] < 2:
        return 0.0

    # Covariance between portfolio and benchmark
    cov_matrix = np.cov(
        df["portfolio"],
        df["benchmark"]
    )

    covariance = cov_matrix[0, 1]

    # Variance of benchmark (market risk)
    benchmark_variance = cov_matrix[1, 1]

    if benchmark_variance == 0:
        return 0.0

    beta = covariance / benchmark_variance

    return beta


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
