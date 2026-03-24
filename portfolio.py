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
    """Download benchmark close, returns, cumulative index.
    Always returns a timezone-naive plain Series regardless of yfinance version.
    """
    adjusted_start = start_date - pd.Timedelta(days=1)

    raw_bench = yf.download(
        symbol,
        start=adjusted_start,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    # ✅ Squeeze: yfinance sometimes returns a 1-col DataFrame instead of Series
    if isinstance(raw_bench, pd.DataFrame):
        raw_bench = raw_bench.squeeze()

    # ✅ Strip timezone: single-ticker downloads may return tz-aware index
    # which causes pd.concat to fail or produce NaNs vs tz-naive portfolio returns
    if hasattr(raw_bench.index, "tz") and raw_bench.index.tz is not None:
        raw_bench.index = raw_bench.index.tz_localize(None)

    bench_ret = raw_bench.pct_change().dropna()
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


# -------- MCTR / Risk Attribution --------

def compute_mctr(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame,
    tickers: list,
    ticker_names: dict,
    cash_pct: float = 0.0
) -> pd.DataFrame:
    """
    Compute Marginal Contribution to Risk (MCTR) and % Risk Contribution
    for each asset in the portfolio.

    Parameters
    ----------
    weights      : equity weight fractions (already scaled by 1-cash_pct, sum = 1-cash_pct)
    cov_matrix   : annualised covariance matrix of equity returns (DataFrame)
    tickers      : list of equity tickers (matches weights order)
    ticker_names : dict {ticker: company_name}
    cash_pct     : cash fraction of total portfolio (0.0 – 1.0)

    Returns
    -------
    DataFrame with columns:
        Company, Ticker, Weight (%), MCTR (%), Risk Contrib (%), Abs Risk Contrib (%)
    """
    w = np.array(weights)

    # Normalise equity weights to sum=1 for covariance math
    equity_total = w.sum()
    if equity_total <= 0:
        return pd.DataFrame()
    w_norm = w / equity_total  # pure equity proportions

    cov = cov_matrix.values  # numpy array

    # Portfolio volatility (equity only, annualised)
    port_var = float(w_norm @ cov @ w_norm)
    port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10

    # Marginal contribution to risk: ∂σ/∂w_i = (Σw)_i / σ
    marginal = (cov @ w_norm) / port_vol  # shape (n,)

    # Absolute risk contribution: w_i × MCTR_i  (sums to port_vol)
    abs_contrib = w_norm * marginal

    # % risk contribution: each asset's share of total portfolio vol
    pct_contrib = (abs_contrib / port_vol) * 100  # sums to 100%

    rows = []
    for i, ticker in enumerate(tickers):
        # Scale weight back to total-portfolio % (include cash)
        total_weight_pct = w[i] * 100  # already scaled e.g. 0.095 → 9.5%
        rows.append({
            "Company": ticker_names.get(ticker, ticker),
            "Ticker": ticker,
            "Weight (%)": round(total_weight_pct, 2),
            "MCTR (%)": round(marginal[i] * 100, 4),
            "% Risk Contribution": round(pct_contrib[i], 2),
            "Abs Risk Contrib (%)": round(abs_contrib[i] * 100, 4),
        })

    df = pd.DataFrame(rows)

    # Add cash row — zero contribution to risk
    if cash_pct > 0:
        cash_row = {
            "Company": "💵 Cash (Risk-Free)",
            "Ticker": "CASH",
            "Weight (%)": round(cash_pct * 100, 2),
            "MCTR (%)": 0.0,
            "% Risk Contribution": 0.0,
            "Abs Risk Contrib (%)": 0.0,
        }
        df = pd.concat([df, pd.DataFrame([cash_row])], ignore_index=True)

    return df


def diversification_ratio(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame
) -> float:
    """
    Diversification Ratio = weighted avg of individual vols / portfolio vol.
    DR > 1 means diversification is reducing risk. Higher = better diversified.
    """
    w = np.array(weights)
    equity_total = w.sum()
    if equity_total <= 0:
        return 1.0
    w_norm = w / equity_total

    individual_vols = np.sqrt(np.diag(cov_matrix.values))
    weighted_avg_vol = float(w_norm @ individual_vols)

    port_vol = np.sqrt(float(w_norm @ cov_matrix.values @ w_norm))
    if port_vol <= 0:
        return 1.0

    return weighted_avg_vol / port_vol


# -------- Active Return Attribution --------

def brinson_attribution(
    returns_data: pd.DataFrame,
    weights: np.ndarray,
    benchmark_returns: pd.Series,
    tickers: list,
    ticker_names: dict,
    cash_pct: float = 0.0,
    risk_free_rate: float = 0.06,
) -> dict:
    """
    Active Return Attribution at individual stock level.

    Uses GEOMETRIC returns and distributes the compounding residual 
    so that the sum of individual contributions exactly matches 
    the true Geometric Active Return (Portfolio Return - Benchmark Return).
    """
    # Defensive squeeze + timezone strip for benchmark
    if isinstance(benchmark_returns, pd.DataFrame):
        benchmark_returns = benchmark_returns.squeeze()
    if hasattr(benchmark_returns.index, "tz") and benchmark_returns.index.tz is not None:
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)

    # BUG 5 FIX: multi-ticker yfinance downloads can return a tz-aware index.
    # pd.concat of tz-aware + tz-naive silently produces an empty / all-NaN
    # DataFrame, breaking all attribution math.  Strip it here to match benchmark.
    if hasattr(returns_data.index, "tz") and returns_data.index.tz is not None:
        returns_data.index = returns_data.index.tz_localize(None)

    # Align all series to common dates
    aligned = pd.concat(
        [returns_data[tickers], benchmark_returns.rename("__bench__")],
        axis=1
    ).dropna()

    stock_rets = aligned[tickers]
    bench_daily = aligned["__bench__"]

    w = np.array(weights)  # sums to (1 - cash_pct)
    num_trading_days = len(aligned)
    daily_rf = risk_free_rate / 252

    # ── True Geometric Returns (compounded) ──────────────────────────────
    geo_bench = float((1 + bench_daily).prod() - 1)
    geo_stock = {ticker: float((1 + stock_rets[ticker]).prod() - 1) for ticker in tickers}
    geo_cash = (1 + daily_rf) ** num_trading_days - 1

    # Blended geometric portfolio return
    blended_daily = stock_rets.dot(w) + cash_pct * daily_rf
    geo_portfolio = float((1 + blended_daily).prod() - 1)
    
    # True Geometric Active Return
    geo_active = geo_portfolio - geo_bench

    # ── Base Contributions & Compounding Residual ────────────────────────
    # Calculate base unadjusted contributions using geometric returns
    unadjusted_contribs = {ticker: w[i] * (geo_stock[ticker] - geo_bench) for i, ticker in enumerate(tickers)}
    cash_unadjusted = cash_pct * (geo_cash - geo_bench) if cash_pct > 0 else 0.0

    total_unadjusted = sum(unadjusted_contribs.values()) + cash_unadjusted
    
    # The compounding residual is the difference between true geo active return 
    # and the sum of unadjusted parts. We distribute this proportionally by weight.
    compounding_residual = geo_active - total_unadjusted

    rows = []
    for i, ticker in enumerate(tickers):
        w_i = float(w[i])
        geo_R_i = geo_stock[ticker]
        
        # Adjusted contribution = Base + (Weight * Residual)
        active_contrib = unadjusted_contribs[ticker] + (w_i * compounding_residual)

        rows.append({
            "Company": ticker_names.get(ticker, ticker),
            "Ticker": ticker,
            "Portfolio Wt (%)": round(w_i * 100, 2),
            "Stock Return (%)": round(geo_R_i * 100, 2),
            "Benchmark Ret (%)": round(geo_bench * 100, 2),
            "Excess Return (%)": round((geo_R_i - geo_bench) * 100, 2),
            "Active Contribution (%)": round(active_contrib * 100, 4),
            "Role": "🟢 Booster" if geo_R_i >= geo_bench else "🔴 Drag",
        })

    stock_df = pd.DataFrame(rows)

    # ── Cash row ──────────────────────────────────────────────────────────
    if cash_pct > 0:
        cash_contrib = cash_unadjusted + (cash_pct * compounding_residual)
        cash_row = {
            "Company":                 "💵 Cash (Risk-Free)",
            "Ticker":                  "CASH",
            "Portfolio Wt (%)":        round(cash_pct * 100, 2),
            "Stock Return (%)":        round(geo_cash * 100, 2),
            "Benchmark Ret (%)":       round(geo_bench * 100, 2),
            "Excess Return (%)":       round((geo_cash - geo_bench) * 100, 2),
            "Active Contribution (%)": round(cash_contrib * 100, 4),
            "Role":                    "🟢 Booster" if geo_cash >= geo_bench else "🔴 Drag",
        }
        stock_df = pd.concat([stock_df, pd.DataFrame([cash_row])], ignore_index=True)

    return {
        "stock_df":             stock_df,
        "active_return_geo":    round(geo_active * 100, 4),
        "portfolio_return":     round(geo_portfolio * 100, 4),
        "benchmark_return":     round(geo_bench * 100, 4),
    }


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