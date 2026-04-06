#app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import re
from auth import *
from portfolio import *
from optimizer import *
from config import *
from AI_insights import *
from pdf_export import generate_portfolio_pdf

st.set_page_config(layout="wide")

# ============================================
# HELPER: TICKER TO COMPANY NAME MAPPING (with Rate Limit Handling)
# ============================================
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_name_single(ticker: str) -> str:
    """
    Fetch a single company name with exponential backoff retry.
    Returns the company name or ticker if fetch fails.
    """
    import time
    
    max_retries = 3
    base_delay = 0.5  # Start with 500ms delay
    
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try different name fields (yfinance inconsistent)
            name = (
                info.get('longName') or 
                info.get('shortName') or 
                ticker
            )
            
            # Shorten very long names
            if len(name) > 30:
                name = name[:27] + "..."
            
            return name
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for rate limit errors
            if "429" in str(e) or "rate" in error_msg or "throttle" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.5s, 1s, 2s
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
            
            # For other errors or final attempt, return ticker as fallback
            return ticker
    
    # All retries exhausted
    return ticker


def get_stock_names(tickers_list):
    """
    Fetch company names for multiple tickers with rate limit handling.
    Uses individual caching per ticker to avoid total failure.
    Returns dict: {ticker: name}
    """
    ticker_to_name = {}
    
    for ticker in tickers_list:
        # Call cached function for each ticker (benefits from cache)
        ticker_to_name[ticker] = get_stock_name_single(ticker)
    
    return ticker_to_name

# ============================================
# INITIALIZE SERVICES
# ============================================
cookies = init_cookie_manager()
users_ws, portfolios_ws = init_google_sheets()
st.session_state["users_ws"] = users_ws
users_df = fetch_users_data(users_ws)

# ============================================
# SESSION STATE
# ============================================
if "signed_up" not in st.session_state:
    st.session_state["signed_up"] = False

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

if "use_optimized" not in st.session_state:
    st.session_state["use_optimized"] = False

if "opt_weights" not in st.session_state:
    st.session_state["opt_weights"] = None

if "loaded_tickers" not in st.session_state:
    st.session_state["loaded_tickers"] = None

if "loaded_weights" not in st.session_state:
    st.session_state["loaded_weights"] = None

# Persistent portfolio state
if "current_tickers" not in st.session_state:
    st.session_state["current_tickers"] = "500034.BO,500520.bo,INFY.NS, ITC.NS, SBIN.NS, DRREDDY.NS,500325.BO, 500696.BO"

if "current_weights_pct" not in st.session_state:
    st.session_state["current_weights_pct"] = None

if "ai_quick_summary" not in st.session_state:
    st.session_state["ai_quick_summary"] = None

if "ai_full_analysis" not in st.session_state:
    st.session_state["ai_full_analysis"] = None

if "ai_analysis_generated" not in st.session_state:
    st.session_state["ai_analysis_generated"] = False

# ============================================
# SIGNUP GATE (BEFORE ANYTHING ELSE)
# ============================================
if not st.session_state["signed_up"]:
    show_signup_gate(users_df, users_ws, cookies)

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("📊 Portfolio Performance & Risk Dashboard")
st.write("Analyze returns, risk, and benchmark performance at one place.")

st.sidebar.header("Portfolio Inputs")

# Ticker input - reads from loaded portfolio
# Determine ticker input value
if st.session_state["loaded_tickers"] is not None:
    default_tickers = ",".join(st.session_state["loaded_tickers"])
    st.session_state["loaded_tickers"] = None
    st.session_state["current_tickers"] = default_tickers
else:
    default_tickers = st.session_state["current_tickers"]

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    value=default_tickers
)

st.session_state["current_tickers"] = tickers
import datetime

# Get today's date
today = datetime.date.today()

# Date inputs with max_value constraint
start_date = st.sidebar.date_input(
    "Start Date", 
    value=pd.to_datetime("2018-01-01").date(),
    max_value=today  # ✅ Cannot select future
)

end_date = st.sidebar.date_input(
    "End Date", 
    value=today,
    min_value=start_date,  # ✅ Must be after start_date
    max_value=today  # ✅ Cannot select future
)

# ✅ Additional validation
if end_date < start_date:
    st.sidebar.error("❌ End date must be after start date")
    st.stop()

if end_date == start_date:
    st.sidebar.warning("⚠️ Start and end dates are the same. Need at least 2 days of data.")
    st.stop()

# ✅ Check minimum data range
date_diff = (end_date - start_date).days
if date_diff < 30:
    st.sidebar.warning(f"⚠️ Only {date_diff} days selected. Recommend at least 30 days for meaningful analysis.")

def validate_ticker(ticker: str) -> bool:
    """Basic ticker validation"""
    return bool(re.match(r'^[A-Z0-9\.\-]{1,10}$', ticker.upper()))

tickers_list = [t.strip().upper() for t in tickers.split(",")]
invalid = [t for t in tickers_list if not validate_ticker(t)]
if invalid:
    st.error(f"Invalid tickers: {', '.join(invalid)}")
    st.stop()
# ============================================
# CASH COMPONENT SESSION STATE
# ============================================
if "cash_weight_pct" not in st.session_state:
    st.session_state["cash_weight_pct"] = 0.0

# ✅ Detect if ticker count changed - reset to equal weights (including cash slot)
if "last_ticker_count" not in st.session_state:
    st.session_state["last_ticker_count"] = len(tickers_list)

if len(tickers_list) != st.session_state["last_ticker_count"]:
    st.session_state["last_ticker_count"] = len(tickers_list)
    st.session_state["current_weights_pct"] = None  # Force equal weights reset
    st.session_state["cash_weight_pct"] = 0.0       # Reset cash too

st.sidebar.info("""Input Guide for Indian Stocks - NSE stocks → (Ticker + `.NS(NSE) or .BO(BSE)`) , e.g. `500325.BO(Reliance)`, `INFY.NS`, `TCS.NS`.""")

# ============================================
# WEIGHT INPUTS (Equities + Cash)
# ============================================
st.sidebar.subheader("🎚 Portfolio Weights (in %)")

# --- Total slots = num tickers + 1 (cash) ---
# Equal weight default splits 100% across all tickers + cash slot
# Cash default is 0, so equal weight only kicks in for equities unless cash was set

weights = []
for i, t in enumerate(tickers_list):
    # Priority: loaded portfolio > stored weights (if count matches) > equal weight
    if st.session_state["loaded_weights"] is not None and i < len(st.session_state["loaded_weights"]):
        default_weight = st.session_state["loaded_weights"][i] * 100
    elif (
        st.session_state["current_weights_pct"]
        and len(st.session_state["current_weights_pct"]) == len(tickers_list)
    ):
        default_weight = st.session_state["current_weights_pct"][i]
    else:
        default_weight = round(100 / len(tickers_list), 2)

    w = st.sidebar.number_input(
        f"Weight for {t} (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(default_weight),
        step=0.5,
    )
    weights.append(w)

# ✅ V2: Cash as a weight input — same style as equity weights
cash_weight_pct = st.sidebar.number_input(
    "💵 Weight for CASH (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state["cash_weight_pct"]),
    step=0.5,
    help="Cash earns the risk-free rate. Include in total weight like any other position."
)
st.session_state["cash_weight_pct"] = cash_weight_pct

# ============================================
# WEIGHT VALIDATION (equities + cash must = 100%)
# ============================================
total_with_cash = sum(weights) + cash_weight_pct

if abs(total_with_cash - 100) > 0.1:
    st.sidebar.warning(
        f"⚠️ Weights sum to {total_with_cash:.2f}%. "
        f"Adjust equities + cash to equal exactly 100%."
    )
elif abs(total_with_cash - 100) > 0.01:
    # Close enough — auto-normalize equities (keep cash fixed)
    equity_total = sum(weights)
    if equity_total > 0:
        scale = (100 - cash_weight_pct) / equity_total
        weights = [w * scale for w in weights]
    st.sidebar.caption("✅ Normalized to 100%")

if cash_weight_pct > 0:
    st.sidebar.caption(
        f"💵 {cash_weight_pct:.1f}% cash earning risk-free rate | "
        f"{100 - cash_weight_pct:.1f}% equities"
    )

# ============================================
# RISK-FREE RATE
# ============================================
risk_free = st.sidebar.number_input(
    "Risk-free rate (%)",
    value=6.0,
    step=0.25
) / 100

st.sidebar.subheader("Optimizer Settings")
lower_limit = st.sidebar.number_input(
    "Min weight per asset (%)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
)

# Store equity weights (as pct) for persistence
st.session_state["current_weights_pct"] = weights.copy()

if st.session_state["loaded_weights"] is not None:
    st.session_state["loaded_weights"] = None

# Convert equity weights to fractions
weights = [w / 100 for w in weights]

# cash_pct as a fraction for downstream use
cash_pct = cash_weight_pct / 100.0

# ============================================
# OPTIMIZER OVERRIDE
# ============================================
if st.session_state["use_optimized"] and st.session_state["opt_weights"] is not None:
    if len(st.session_state["opt_weights"]) == len(tickers_list):
        weights = st.session_state["opt_weights"]
        st.sidebar.success("✅ Using Optimized Weights")

        if st.sidebar.button("🔄 Back to Manual Weights"):
            st.session_state["use_optimized"] = False
            st.rerun()
    else:
        st.session_state["use_optimized"] = False
        st.sidebar.info("ℹ️ Using Manual Weights")
else:
    st.sidebar.info("ℹ️ Using Manual Weights")

# ============================================
# FINAL WEIGHT CHECK (equities only, post-override)
# ============================================
total = sum(weights)

if abs(total - 1) > 0.001 and cash_pct == 0:
    # Only show pure equity warning when no cash is set (combined check already done above)
    pass
elif abs(total - 1) > 1e-6 and cash_pct == 0:
    st.sidebar.caption("✅ Total: 100%")

# ============================================
# DOWNLOAD DATA
# ============================================
# Fetch 1 extra day before to avoid losing Day 1 after pct_change
adjusted_start = start_date - pd.Timedelta(days=1)

raw = yf.download(
    tickers_list,
    start=adjusted_start,  # Start 1 day earlier
    end=end_date,
    auto_adjust=True,
    progress=False
)


# ============================================
# TABS
# ============================================
tab_overview, tab_risk, tab_benchmark, tab_monte, tab_optimizer, tab_frontier, tab_mctr, tab_return_attr, tab_ai = st.tabs(
    ["📊 Overview", "⚠️ Risk", "📈 Benchmark", "🎲 Monte Carlo", "🔧 Optimizer", "🏔️ Efficient Frontier", "📐 Risk Attribution", "📊 Return Attribution", "🤖 AI Analysis"]
)
# ============================================
# DATA & RETURNS (computed outside tabs — shared by all tabs)
# ============================================
# ============================================
# DATA & RETURNS (computed outside tabs — shared by all tabs)
# ============================================
data = raw["Close"]
if "Adj Close" in raw.columns:
    data = raw["Adj Close"]
data = data.dropna()

# BUG 1 FIX: Compute returns FIRST on full data (including adjusted_start day),
# THEN filter to requested start_date — prevents losing the first 2 trading days.
returns_full = data.pct_change().dropna()
returns = returns_full[returns_full.index >= pd.to_datetime(start_date)]

# BUG 2 FIX: Handle single-ticker (yfinance returns a Series, not a DataFrame).
# Also enforce user's ticker order (yfinance sorts columns alphabetically).
if isinstance(returns, pd.DataFrame) and len(tickers_list) > 1:
    returns = returns[tickers_list]
elif isinstance(returns, pd.Series):
    returns = returns.to_frame(name=tickers_list[0])

# Keep data aligned to the same window as returns for price charts
data = data[data.index >= pd.to_datetime(start_date)]

# Ticker names (cached)
ticker_names = get_stock_names(tickers_list)

# ============================================
# OVERVIEW TAB
# ============================================
# ============================================
# OVERVIEW TAB
# ============================================
with tab_overview:
    st.subheader("📈 Price History")

    # ✅ V2: Cash component banner
    if cash_pct > 0:
        st.info(f"💵 **Cash Component:** {cash_pct*100:.1f}% in cash + {(1-cash_pct)*100:.1f}% in equities below.")

    # Prepare chart data
    price_df = data.reset_index().melt('Date', var_name='Ticker', value_name='Price')
    price_df['Company'] = price_df['Ticker'].map(ticker_names)

    chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X('Date:T', axis=alt.Axis(format='%b %Y', labelAngle=0, labelOverlap=True)),
            y=alt.Y('Price:Q', title="Price"),
            color=alt.Color('Company:N', legend=alt.Legend(
                orient='bottom', 
                title=None,
                columns=6,
                symbolType='stroke'
            )),
            tooltip=['Date:T', 'Company:N', 'Ticker:N', 'Price:Q']
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    # 2. Last 5 Days Prices Table
    st.subheader("📊 Last 5 Days: Closing Prices")
    price_display = data.tail().copy()
    price_display.columns = [ticker_names.get(t, t) for t in price_display.columns]
    st.dataframe(price_display, use_container_width=True)
    st.subheader("📊 Latest Returns")

    if not returns.empty:
        returns_display = returns.tail().copy()
        returns_display.columns = [ticker_names.get(t, t) for t in returns_display.columns]
        returns_formatted = returns_display.style.format(lambda x: f"{x:.2%}")
        st.dataframe(returns_formatted, use_container_width=True)
    else:
        st.warning("No price data available for the selected tickers/date range.")
        st.stop()

if returns.empty:
    st.error("No price data available for the selected tickers/date range.")
    st.stop()

if returns.shape[1] != len(weights):
    weights = np.repeat(1/returns.shape[1], returns.shape[1])

# ============================================
# CASH COMPONENT BLENDING
# ============================================
# cash_pct is already a fraction (e.g. 0.20 for 20%)
# weights are equity fractions that should already sum to (1 - cash_pct)
# We use them directly without additional scaling
daily_rf = risk_free / 252
equity_return = returns.dot(np.array(weights))
portfolio_return = equity_return + cash_pct * daily_rf
cumulative = (1 + portfolio_return).cumprod()


# ============================================
# RISK TAB
# ============================================
with tab_risk:
    st.header("📉 Portfolio Risk Metrics")

    # ✅ V2: Cash component info banner
    if cash_pct > 0:
        st.info(
            f"💵 **Cash Component Active:** {cash_pct*100:.1f}% allocated to cash (earning {risk_free*100:.2f}% p.a. risk-free rate). "
            f"Metrics below reflect the blended portfolio of {(1-cash_pct)*100:.1f}% equities + {cash_pct*100:.1f}% cash."
        )

    metrics = risk_metrics(portfolio_return, cumulative, risk_free)

    # --- Core Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility (Annualized)", f"{metrics['volatility_pct']:.2f}%")
    col1.caption("📊 Measures how **choppy** returns are. Higher = riskier.")

    col2.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    col2.caption("⚖️ Return earned **per unit of risk**. >1 is strong.")

    col3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    # Compute peak → trough dates for the max drawdown period
    _rolling_max = cumulative.cummax()
    _dd_series = (cumulative - _rolling_max) / _rolling_max
    _trough_date = _dd_series.idxmin()
    _peak_date   = cumulative.loc[:_trough_date].idxmax()
    col3.caption(
        f"📉 Worst peak‑to‑trough fall – shows **pain during crashes**. "
        f"Peak: {_peak_date.strftime('%d %b %Y')} → Trough: {_trough_date.strftime('%d %b %Y')}"
    )

    col4.metric("Sortino Ratio", f"{metrics['sortino']:.2f}" if not np.isnan(metrics["sortino"]) else "—")
    col4.caption("🎯 Risk‑adjusted return using **downside volatility only**.")

    st.divider()

    # --- Tail Risk Metrics ---
    st.subheader("📉 Downside & Tail Risk Metrics")

    row1_col1, row1_col2 = st.columns(2)
    row1_col1.metric("VaR (95%)", f"{metrics['var_95_pct']:.2f}%")
    row1_col1.caption("🛡️ Expected worst daily loss on **95% of days**.")

    row1_col2.metric("CVaR (95%)", f"{metrics['cvar_95_pct']:.2f}%")
    row1_col2.caption("🔥 Average loss **when VaR is breached** (tail risk).")

    row2_col1, row2_col2 = st.columns(2)
    row2_col1.metric("VaR (99%)", f"{metrics['var_99_pct']:.2f}%")
    row2_col1.caption("⚠️ Extreme loss threshold (1% worst days).")

    row2_col2.metric("CVaR (99%)", f"{metrics['cvar_99_pct']:.2f}%")
    row2_col2.caption("💥 Expected loss during **extreme market stress**.")



    with st.expander("ℹ️ What do Sortino Ratio, VaR & CVaR indicate?"):
        st.markdown("""
    **Sortino Ratio**  
    Measures return per unit of **downside risk**, ignoring upside volatility.
    Preferred over Sharpe when evaluating real-world portfolios.

    **VaR (Value at Risk)**  
    Estimates how much you might lose on a bad day, with a given confidence level.

    **CVaR / TVaR (Conditional VaR)**  
    Goes one step further — it measures how bad losses are **when VaR is exceeded**.
    This is critical for understanding **tail risk**.
        """)



    # --- Correlation Matrix Heatmap (unchanged) ---
    st.subheader("📊 Correlation matrix of assets")
    st.caption("Lower correlation improves diversification and can reduce portfolio volatility.")

    if returns.shape[1] < 2:
        st.info("Add at least two tickers to view correlations.")
    else:
        # Compute correlation
        corr_matrix = returns.corr()

        # Name the index/columns to avoid collisions and reshape safely
        corr_matrix.index.name = "Ticker1"
        corr_matrix.columns.name = "Ticker2"

        # Melt into tidy long form
        corr_df = corr_matrix.reset_index().melt(
            id_vars="Ticker1",
            var_name="Ticker2",
            value_name="Correlation",
        )

        # Get company names
        ticker_names = get_stock_names(corr_matrix.index.tolist())

        # Map tickers to names in correlation dataframe
        corr_df['Company1'] = corr_df['Ticker1'].map(ticker_names)
        corr_df['Company2'] = corr_df['Ticker2'].map(ticker_names)

        # Build heatmap
        # Build heatmap
        heatmap = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "Company1:N", 
                    sort=None, 
                    title=None, 
                    axis=alt.Axis(
                        labelAngle=-45,      # ✅ Slanted for easier reading
                        labelOverlap=False,   # ✅ Ensures no labels are skipped
                        labelLimit=150        # ✅ Prevents long names from cutting off early
                    )
                ),
                y=alt.Y("Company2:N", sort=None, title=None),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="redgrey", domain=[-1, 1]),
                    legend=alt.Legend(title="Correlation")
                ),
                tooltip=[
                    alt.Tooltip("Company1:N", title="Asset A"),
                    alt.Tooltip("Ticker1:N", title="Ticker A"),
                    alt.Tooltip("Company2:N", title="Asset B"),
                    alt.Tooltip("Ticker2:N", title="Ticker B"),
                    alt.Tooltip("Correlation:Q", format=".2f"),
                ],
            )
            .properties(
                width=alt.Step(60), # ✅ Fixed column width ensures labels have room
                height=500, 
                title="Correlation Heatmap"
            )
        )

        # Build labels (the numbers inside the boxes)
        labels = (
            alt.Chart(corr_df)
            .mark_text(baseline="middle", align="center", fontSize=12)
            .encode(
                x="Company1:N",
                y="Company2:N",
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "datum.Correlation > 0.6 || datum.Correlation < -0.6",
                    alt.value("white"),
                    alt.value("black")
                )
            )
        )

        # Combine and apply final axis configuration
        final_chart = (heatmap + labels).configure_axis(
            labelFontSize=11,
            titleFontSize=13
        )

        st.altair_chart(final_chart, use_container_width=True)
    
    st.subheader("🔄 Rolling Risk (Volatility & Sharpe)")
    st.caption("Rolling metrics show **how risk and performance change over time**.")

    st.subheader("🔍 Rolling Window Settings")
    window = st.slider(
        "Rolling window (trading days)",
        min_value=20,
        max_value=252,
        value=60,
        step=5
    )

    # Use helper from portfolio.py
    rolling_vol, rolling_sharpe = rolling_risk(portfolio_return, window, risk_free)

    # Plot rolling volatility
    vol_df = rolling_vol.reset_index()
    vol_df.columns = ["Date", "Rolling Volatility (%)"]
    chart_vol = (
        alt.Chart(vol_df)
        .mark_line(color="#e63946")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format='%b %Y', labelAngle=0)),
            y="Rolling Volatility (%):Q"
        )
        .properties(title="Rolling Annualized Volatility", height=320, width=950)
    )
    st.altair_chart(chart_vol, use_container_width=False)

    # Plot rolling Sharpe
    sharpe_df = rolling_sharpe.reset_index()
    sharpe_df.columns = ["Date", "Rolling Sharpe"]
    chart_sharpe = (
        alt.Chart(sharpe_df)
        .mark_line(color="#8e70e2")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format='%b %Y', labelAngle=0)),
            y="Rolling Sharpe:Q"
        )
        .properties(title="Rolling Sharpe Ratio", height=320, width=950)
    )
    st.altair_chart(chart_sharpe, use_container_width=False)


    with st.expander("🔍 What does Rolling Volatility & Sharpe mean?"):
        st.markdown("""
    **Rolling Volatility:**  
    Shows when portfolio shifted from **calm** to **turbulent**.

    **Rolling Sharpe:**  
    Reveals when the portfolio delivered **good vs poor risk-adjusted returns**.
        """)

# ============================================
# BENCHMARK TAB
# ============================================
# ============================================
# BENCHMARK DATA (shared across tabs)
# ============================================
# Defined outside tabs so both Benchmark tab and Return Attribution tab can use it
benchmark_names = {
    "^NSEI": "Nifty 50",
    "^GSPC": "S&P 500",
    "^NDX": "Nasdaq 100",
    "^BSESN": "Sensex"
}

with tab_benchmark:
    st.header("📊 Portfolio vs Benchmark")

    benchmark_symbol = st.selectbox(
        "Select benchmark index:",
        ("^NSEI", "^GSPC", "^NDX", "^BSESN"),
        index=0
    )

    benchmark_label = benchmark_names.get(benchmark_symbol, "Benchmark")
    # Use helper from portfolio.py instead of inline download
    benchmark_returns, bench_cum = benchmark_series(benchmark_symbol, adjusted_start, end_date)

    # Align Daily Returns FIRST (resets both to start line)
    aligned_df = pd.concat([portfolio_return, benchmark_returns], axis=1).dropna()
    aligned_df.columns = ["Portfolio", benchmark_label]

    # Calculate Cumulative Growth on aligned data (both start at 1.0)
    aligned_cumulative = (1 + aligned_df).cumprod()

    # 4. Calculate Totals & Alpha
    portfolio_total = aligned_cumulative.iloc[-1]["Portfolio"] - 1
    benchmark_total = aligned_cumulative.iloc[-1][benchmark_label] - 1
    alpha = (portfolio_total - benchmark_total) * 100
    color = "green" if alpha >= 0 else "red"

    # --- 2. Prepare Data for Chart ---
    # We use the aligned_cumulative df we created earlier, which ensures
    # both lines start at the exact same point (1.0) on the chart.
    compare_df = aligned_cumulative.reset_index().melt(
        id_vars="Date", 
        var_name="variable", 
        value_name="value"
    )

    # --- 3. Plot Chart ---
    # Note: We removed 'transform_fold' because we already melted the data above
    chart_compare = (
        alt.Chart(compare_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format='%b %Y', labelAngle=0)),
            y=alt.Y("value:Q", title="Growth Index (Start = 1.0)"),
            color=alt.Color("variable:N", legend=alt.Legend(title=None))
        )
        .properties(height=420, width=1000, title="Portfolio vs Benchmark")
    )
    st.altair_chart(chart_compare, use_container_width=False)

    # Extract dates for the label
    start_str = aligned_cumulative.index[0].strftime('%d %b %Y')
    end_str   = aligned_cumulative.index[-1].strftime('%d %b %Y')

    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown(
            f"""
            <h4>📈 Alpha vs {benchmark_label}: 
            <span style='color:{color}'>{alpha:.2f}%</span>
            </h4>
            <p style='font-size: 14px; color: #888; margin-top: -10px; font-weight: 400;'>
                from {start_str} to {end_str}
            </p>
            """,
            unsafe_allow_html=True
        )

    with col_right:
        beta = portfolio_beta(portfolio_return, benchmark_returns)
        st.metric(label="📐 Beta", value=f"{beta:.2f}")



# ============================================
# MONTE CARLO TAB
# ============================================
with tab_monte:
    st.header("🎲 Monte Carlo Simulation")
    st.write("""
    We simulate thousands of possible future paths for this portfolio
    using historical returns and volatility.  
    This helps estimate **best-case, worst-case, and typical outcomes**.
    """)

    daily_returns = portfolio_return
    mu = daily_returns.mean()
    sigma = daily_returns.std()

    st.subheader("Simulation Settings")
    num_days = st.slider(
        "Number of trading days to simulate",
        min_value=60, max_value=756, value=252, step=10
    )

    num_sims = st.slider(
        "Number of simulations",
        min_value=100, max_value=3000, value=200, step=100
    )

    # ✅ Use portfolio.py helper instead of local function
    sim_results = run_simulation(mu, sigma, num_days, num_sims)

    st.subheader("Simulated Future Portfolio Paths")
    sample = min(60, num_sims)
    sim_df = pd.DataFrame(sim_results[:, :sample])
    sim_df["Day"] = np.arange(1, num_days + 1)
    sim_melted = sim_df.melt(id_vars="Day", var_name="Simulation", value_name="Value")

    p5  = np.percentile(sim_results, 5, axis=1)
    p50 = np.percentile(sim_results, 50, axis=1)
    p95 = np.percentile(sim_results, 95, axis=1)

    percentile_df = pd.DataFrame({
        "Day": np.arange(1, num_days + 1),
        "P5": p5, "P50": p50, "P95": p95
    })

    # Charts (unchanged)
    lower_line = alt.Chart(percentile_df).mark_line(color="#d61f1f").encode(
        x="Day:Q", y=alt.Y("P5:Q", axis=alt.Axis(title="Portfolio Value (x)"))
    )
    upper_line = alt.Chart(percentile_df).mark_line(color="#4ade80").encode(
        x="Day:Q", y="P95:Q"
    )
    median_line = alt.Chart(percentile_df).mark_line(color="orange", size=2).encode(
        x="Day:Q", y="P50:Q"
    )
    chart_mc = alt.Chart(sim_melted).mark_line(opacity=0.2).encode(
        x="Day:Q", y="Value:Q"
    ).properties(height=420, width=900)

    final_chart = lower_line + upper_line + chart_mc + median_line
    st.altair_chart(final_chart, use_container_width=False)

    # Summary (unchanged)
    final_values = sim_results[-1, :]
    p5  = np.percentile(final_values, 5)
    p50 = np.percentile(final_values, 50)
    p95 = np.percentile(final_values, 95)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"""
        <div style="padding:14px;border-radius:12px;background-color:#111827;border:1px solid #2d2d2d;">
            <div style="font-size:14px;opacity:0.85;">📉 Bad Case (5th %ile)</div>
            <div style="font-size:30px;font-weight:200;color:#d61f1f;">{p5:.2f}x</div>
        </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
        <div style="padding:14px;border-radius:12px;background-color:#111827;border:1px solid #2d2d2d;">
            <div style="font-size:14px;opacity:0.85;">📊 Median (50th %ile)</div>
            <div style="font-size:30px;font-weight:200;color:orange;">{p50:.2f}x</div>
        </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
        <div style="padding:14px;border-radius:12px;background-color:#111827;border:1px solid #2d2d2d;">
            <div style="font-size:14px;opacity:0.85;">📈 Good Case (95th %ile)</div>
            <div style="font-size:30px;font-weight:200;color:#4ade80;">{p95:.2f}x</div>
        </div>
    """, unsafe_allow_html=True)

    st.write(f"""
    ➡ With 90% confidence, your portfolio may end between **{p5:.2f}x** and **{p95:.2f}x** 
    over the next {num_days} trading days.
    """)
    st.caption("""
    These estimates come from Monte Carlo simulations based on historical volatility 
    and returns. They do **not** guarantee outcomes, but help visualize risk and uncertainty.
    """)

# ============================================
# OPTIMIZER TAB
# ============================================

with tab_optimizer:
    st.header("🔧 Portfolio Optimizer")

    # --- 1. Calculations ---
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # ✅ FIX: Ensure tickers_list matches mean_returns.index order
    tickers_list_sorted = mean_returns.index.tolist()

    # --- 2. Strategy Selection ---
    goal = st.radio(
        "Optimization Goal:",
        ["🚀 Max Sharpe (Growth)", "🛡️ Min Volatility (Safety)"],
        horizontal=True
    )

    if st.button("Run Optimization", type="primary"):
        # ✅ V2: Adjust lower_limit so after equity scaling each asset
        # still meets user's intended min % of total portfolio
        equity_scale = 1.0 - cash_pct
        adjusted_lower_limit_opt = lower_limit / equity_scale if equity_scale > 0 else lower_limit

        # BUG 3 FIX: Catch impossible constraints before scipy silently fails and
        # returns garbage unoptimized weights (result.success = False).
        if adjusted_lower_limit_opt * len(tickers_list_sorted) > 100.0:
            st.error(
                f"⚠️ Cannot optimize: the minimum weight ({lower_limit:.0f}%) × "
                f"{len(tickers_list_sorted)} assets = "
                f"{lower_limit * len(tickers_list_sorted):.0f}%, which exceeds the "
                f"available equity portion ({equity_scale * 100:.1f}%). "
                f"Lower the minimum weight or reduce cash."
            )
            st.stop()

        if "Max Sharpe" in goal:
            opt_weights_calc, ret, vol, sharpe = get_max_sharpe_portfolio(
                mean_returns, cov_matrix, risk_free, adjusted_lower_limit_opt
            )
        else:
            opt_weights_calc, ret, vol, sharpe = get_min_volatility_portfolio(
                mean_returns, cov_matrix, risk_free, adjusted_lower_limit_opt
            )

        # ✅ V2: Scale optimizer output by equity fraction immediately
        # opt_weights_calc sums to 1.0 (pure equity). Scale to (1 - cash_pct) so
        # stored weights + cash always = 100%.
        equity_scale = 1.0 - cash_pct
        opt_weights_scaled_stored = opt_weights_calc * equity_scale

        # Save results — store SCALED weights so Apply button is consistent
        st.session_state["opt_weights"] = opt_weights_scaled_stored.tolist()
        st.session_state["opt_weights_raw"] = opt_weights_calc.tolist()  # keep raw for optimizer stats
        st.session_state["opt_stats"] = (ret, vol, sharpe)
        st.session_state["opt_tickers_order"] = tickers_list_sorted
        st.rerun()

    # --- 3. Display Results ---
    if "opt_weights" in st.session_state and "opt_stats" in st.session_state:
        # ✅ Use raw weights for stats (they must sum to 1 for correct Sharpe/vol math)
        opt_weights_raw = np.array(st.session_state.get("opt_weights_raw", st.session_state["opt_weights"]))
        # Normalise raw in case it was stored before this fix
        if abs(opt_weights_raw.sum() - 1.0) > 0.01:
            opt_weights_raw = opt_weights_raw / opt_weights_raw.sum()

        opt_weights_display = np.array(st.session_state["opt_weights"])  # scaled version for display
        ret, vol, sharpe = st.session_state["opt_stats"]

        # Safety Check: Ticker mismatch
        if opt_weights_display.shape[0] != mean_returns.shape[0]:
            st.warning("⚠️ Ticker list changed. Please click 'Run Optimization' again.")
            opt_weights_display = np.repeat(1/mean_returns.shape[0], mean_returns.shape[0])
            st.session_state["opt_weights"] = opt_weights_display
        else:
            # ✅ Manual weights — normalize to sum=1 for stats (equity portion only)
            raw_manual = np.array(weights)  # may sum to (1 - cash_pct)
            equity_total = raw_manual.sum()
            manual_weights_norm = raw_manual / equity_total if equity_total > 0 else raw_manual

            # Calculate stats using normalized equity weights (optimizer always works on 100% equity)
            opt_ret, opt_vol, opt_sharpe = portfolio_stats(
                opt_weights_raw, mean_returns, cov_matrix, risk_free
            )
            manual_ret, manual_vol, manual_sharpe = portfolio_stats(
                manual_weights_norm, mean_returns, cov_matrix, risk_free
            )

            # --- Comparison Table ---
            st.subheader("📊 Manual vs Optimized Portfolio (Annualized)")
            compare_df = pd.DataFrame({
                "Metric": ["Expected Return", "Volatility (Risk)", "Sharpe Ratio"],
                "Manual": [f"{manual_ret*100:.2f}%", f"{manual_vol*100:.2f}%", f"{manual_sharpe:.2f}"],
                "Optimized": [f"{opt_ret*100:.2f}%", f"{opt_vol*100:.2f}%", f"{opt_sharpe:.2f}"]
            }, index=[1, 2, 3])

            # Convert to strings explicitly to prevent auto-formatting
            compare_df["Manual"] = compare_df["Manual"].astype(str)
            compare_df["Optimized"] = compare_df["Optimized"].astype(str)

            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            # --- Charts ---
            # Cumulative returns chart — use normalized equity weights for both
            # (chart shows equity-only performance for fair comparison)
            manual_daily_ret = returns.dot(manual_weights_norm)
            opt_daily_ret = returns.dot(opt_weights_raw)

            aligned_returns = pd.concat([manual_daily_ret, opt_daily_ret], axis=1).dropna()
            aligned_returns.columns = ["Manual", "Optimized"]

            manual_cum = (1 + aligned_returns["Manual"]).cumprod()
            opt_cum = (1 + aligned_returns["Optimized"]).cumprod()
            perf_df = pd.DataFrame({
                "Date": manual_cum.index,
                "Manual Portfolio": manual_cum,
                "Optimized Portfolio": opt_cum
            }).melt("Date", var_name="Portfolio", value_name="Growth Index")
            perf_chart = alt.Chart(perf_df).mark_line().encode(
                x=alt.X("Date:T", axis=alt.Axis(format='%b %Y', title=None, labelAngle=0)),
                y=alt.Y("Growth Index:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Portfolio:N", scale=alt.Scale(
                    domain=['Manual Portfolio', 'Optimized Portfolio'], 
                    range=['#cccccc', '#4ade80']
                )),
                tooltip=["Date:T", "Portfolio", alt.Tooltip("Growth Index", format=".2f")]
            ).properties(height=350, width=800)
            st.altair_chart(perf_chart, use_container_width=True)
            st.caption("This chart shows **historical backtested performance** based on your chosen period. It is not a forecast.")

            # Get company names
            ticker_names = get_stock_names(tickers_list)

            # ✅ V2: opt_weights_display is already scaled to (1 - cash_pct)
            # manual weights (raw_manual) already sum to (1 - cash_pct) when cash > 0
            alloc_df = pd.DataFrame({
                "Ticker": tickers_list,
                "Company": [ticker_names[t] for t in tickers_list],
                "Manual": raw_manual * 100,
                "Optimized": opt_weights_display * 100
            })

            # Add cash row if applicable
            if cash_pct > 0:
                cash_row = pd.DataFrame([{
                    "Ticker": "CASH",
                    "Company": "💵 Cash (Risk-Free)",
                    "Manual": cash_pct * 100,
                    "Optimized": cash_pct * 100
                }])
                alloc_df = pd.concat([alloc_df, cash_row], ignore_index=True)

            # Melt while keeping both Ticker and Company
            alloc_df_melted = alloc_df.melt(
                id_vars=["Ticker", "Company"], 
                var_name="Type", 
                value_name="Weight (%)"
            )

            alloc_chart = alt.Chart(alloc_df_melted).mark_bar().encode(
            x=alt.X(
                "Company:N", 
                title=None, 
                sort=None, 
                axis=alt.Axis(
                    labelAngle=-45,
                    labelOverlap=False,
                    labelLimit=150
                )
            ),
            y=alt.Y("Weight (%):Q", title="Weight (%)"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=['Manual', 'Optimized'], 
                range=['#cccccc', '#4ade80']
            )),
            xOffset="Type:N",
            tooltip=["Company:N", "Ticker:N", "Type:N", alt.Tooltip("Weight (%)", format=".1f")]
        ).properties(
            height=350,
            width=alt.Step(80)
        )
                        
            st.altair_chart(alloc_chart, use_container_width=True)

            # weights table
            st.subheader("Optimized Portfolio Weights")
            if cash_pct > 0:
                st.caption(f"✅ Equity weights scaled to {(1-cash_pct)*100:.1f}% of portfolio. Cash ({cash_pct*100:.1f}%) is fixed.")

            # Get company names
            ticker_names = get_stock_names(mean_returns.index.tolist())

            # opt_weights_display is already scaled by (1 - cash_pct)
            opt_df = pd.DataFrame({
                "Company": [ticker_names[t] for t in mean_returns.index],
                "Ticker": list(mean_returns.index),
                "Weight %": (opt_weights_display * 100).round(2)
            })

            # Add cash row
            if cash_pct > 0:
                cash_df_row = pd.DataFrame([{
                    "Company": "💵 Cash (Risk-Free)",
                    "Ticker": "CASH",
                    "Weight %": round(cash_pct * 100, 2)
                }])
                opt_df = pd.concat([opt_df, cash_df_row], ignore_index=True)

            # Total row
            total_row = pd.DataFrame([{
                "Company": "TOTAL",
                "Ticker": "—",
                "Weight %": round(opt_df["Weight %"].sum(), 2)
            }])
            opt_df = pd.concat([opt_df, total_row], ignore_index=True)

            st.dataframe(opt_df, hide_index=True, use_container_width=True)

            # apply button — opt_weights_display already sums to (1 - cash_pct)
            # so when weights is set to this, equity + cash = 100% ✅
            if st.button("✅ Apply These Weights", type="primary"):
                st.session_state["use_optimized"] = True
                st.session_state["opt_weights"] = opt_weights_display.tolist()
                st.success("✅ Optimized weights applied!")
                st.toast("✅ Optimized weights applied!", icon="✅")
                st.rerun()


# ============================================
# EFFICIENT FRONTIER TAB
# ============================================
with tab_frontier:
    st.header("🏔️ Efficient Frontier")
    
    st.write("""
    The **Efficient Frontier** shows all possible portfolio combinations and identifies 
    the optimal risk-return trade-offs. Each dot represents a different allocation strategy.
    """)
    
        # ✅ SAFETY: Clear frontier cache if tickers changed
    if 'frontier_data' in st.session_state:
        # ✅ Check both count AND ticker names
        cached_tickers = st.session_state.get('frontier_tickers')
        current_tickers = ",".join(sorted(tickers_list))
        
        if cached_tickers != current_tickers:
            del st.session_state['frontier_data']
            if 'frontier_tickers' in st.session_state:
                del st.session_state['frontier_tickers']
            st.info("ℹ️ Ticker list changed. Please regenerate the efficient frontier.")
        
    # === USER CONTROLS ===
    st.subheader("⚙️ Simulation Settings")
    
    col_settings1, col_settings2 = st.columns(2)
    
    
    with col_settings1:
        num_random = st.slider(
            "Number of random portfolios",
            min_value=500,
            max_value=5000,
            value=2000,
            step=500,
            help="More portfolios = smoother visualization but slower computation"
        )
    
    with col_settings2:
        target_risk = st.slider(
            "Your risk tolerance (annual volatility %)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
            help="Maximum volatility you're comfortable with"
        )

    # === GENERATE BUTTON ===
    if st.button("🚀 Generate Efficient Frontier", type="primary", use_container_width=True):
        with st.spinner("🔄 Generating portfolios... This may take 10-30 seconds..."):
            
            # Calculate required data
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252

            # ✅ V2: Adjust lower_limit so that after equity scaling, each asset
            # still meets the user's intended minimum % of total portfolio.
            # e.g. user wants 5% min, cash=5% → equity fraction=0.95
            # optimizer must enforce 5/0.95 = 5.26% so after ×0.95 → 5.0% ✅
            equity_scale_frontier = 1.0 - cash_pct
            adjusted_lower_limit = lower_limit / equity_scale_frontier if equity_scale_frontier > 0 else lower_limit

            # Generate random portfolios (the dots)
            random_df = generate_random_portfolios(
                mean_returns,
                cov_matrix,
                risk_free,
                num_portfolios=num_random,
                lower_limit=adjusted_lower_limit
            )
            
            # Generate efficient frontier curve (the red line)
            frontier_df = get_efficient_frontier_curve(
                mean_returns,
                cov_matrix,
                risk_free,
                lower_limit=adjusted_lower_limit,
                num_points=50
            )
                        # Calculate current portfolio position
            current_weights = np.array(weights)

            # ✅ FIX: Get ticker order from returns dataframe (matches mean_returns order)
            tickers_for_display = returns.columns.tolist()

            current_return = portfolio_return_opt(current_weights, mean_returns)
            current_vol = portfolio_volatility_opt(current_weights, cov_matrix)
            current_sharpe = (current_return - risk_free) / current_vol if current_vol != 0 else 0.0
            
            # Find top 5 efficient portfolios
            top5_efficient = frontier_df.nlargest(5, 'sharpe')
            
            # Find optimal portfolio within risk tolerance
            optimal_at_risk = find_optimal_portfolio_at_risk(frontier_df, target_risk)
            
            st.session_state['frontier_data'] = {
                'random': random_df,
                'frontier': frontier_df,
                'top5': top5_efficient,
                'optimal_at_risk': optimal_at_risk,
                'current': {
                    'return': current_return,
                    'volatility': current_vol,
                    'sharpe': current_sharpe,
                    'weights': current_weights
                },
                'params': {
                    'num_random': num_random,
                    'target_risk': target_risk
                }
            }

# ✅ Store ticker snapshot for cache validation
            st.session_state['frontier_tickers'] = ",".join(sorted(tickers_list))
            
            st.success("✅ Efficient Frontier generated successfully!")
            st.rerun()
    
    # === DISPLAY RESULTS (OUTSIDE BUTTON BLOCK) ===
    if 'frontier_data' in st.session_state:
        data = st.session_state['frontier_data']
        random_df = data['random']
        frontier_df = data['frontier']
        top5 = data['top5']
        optimal_at_risk = data['optimal_at_risk']
        current = data['current']
        params = data['params']
        
        # === CHART ===
        st.subheader("📊 Risk-Return Landscape")
        
        # Prepare data for visualization
        random_df_chart = random_df.copy()
        random_df_chart['type'] = 'Random Portfolio'
        random_df_chart['return_pct'] = random_df_chart['return'] * 100
        random_df_chart['volatility_pct'] = random_df_chart['volatility'] * 100
        
        frontier_df_chart = frontier_df.copy()
        frontier_df_chart['type'] = 'Efficient Frontier'
        frontier_df_chart['return_pct'] = frontier_df_chart['return'] * 100
        frontier_df_chart['volatility_pct'] = frontier_df_chart['volatility'] * 100
        
        current_df = pd.DataFrame([{
            'volatility_pct': current['volatility'] * 100,
            'return_pct': current['return'] * 100,
            'sharpe': current['sharpe'],
            'type': 'Your Portfolio'
        }])

        # Random portfolios (gray dots)
        scatter = alt.Chart(random_df_chart).mark_circle(
            size=30,
            opacity=0.3
        ).encode(
            x=alt.X('volatility_pct:Q', 
                   title='Risk (Annualized Volatility %)',
                   scale=alt.Scale(zero=False)),
            y=alt.Y('return_pct:Q', 
                   title='Expected Return (%)',
                   scale=alt.Scale(zero=False)),
            color=alt.Color('sharpe:Q',
                          scale=alt.Scale(scheme='viridis'),
                          legend=alt.Legend(title='Sharpe Ratio')),
            tooltip=[
                alt.Tooltip('volatility_pct:Q', title='Risk (%)', format='.2f'),
                alt.Tooltip('return_pct:Q', title='Return (%)', format='.2f'),
                alt.Tooltip('sharpe:Q', title='Sharpe', format='.2f')
            ]
        ).properties(
            width=800,
            height=500
        )

        # Efficient frontier (red dashed line)
        frontier_line = alt.Chart(frontier_df_chart).mark_line(
            color='#FF4444',
            size=3,
            strokeDash=[5, 5]
        ).encode(
            x='volatility_pct:Q',
            y='return_pct:Q',
            tooltip=[
                alt.Tooltip('volatility_pct:Q', title='Risk (%)', format='.2f'),
                alt.Tooltip('return_pct:Q', title='Return (%)', format='.2f'),
                alt.Tooltip('sharpe:Q', title='Sharpe', format='.2f')
            ]
        )

        # Your current portfolio (green diamond) - ✅ SOFTER GREEN
        current_point = alt.Chart(current_df).mark_point(
            shape='diamond',
            size=400,
            color='#2ECC71',  # ✅ Changed from #00FF00
            filled=True,
            stroke='white',
            strokeWidth=3
        ).encode(
            x='volatility_pct:Q',
            y='return_pct:Q',
            tooltip=[
                alt.Tooltip('volatility_pct:Q', title='Your Risk (%)', format='.2f'),
                alt.Tooltip('return_pct:Q', title='Your Return (%)', format='.2f'),
                alt.Tooltip('sharpe:Q', title='Your Sharpe', format='.2f')
            ]
        )

        # Risk tolerance vertical line (orange)
        risk_line_data = pd.DataFrame({
            'volatility_pct': [params['target_risk'], params['target_risk']],
            'return_pct': [
                random_df_chart['return_pct'].min(),
                random_df_chart['return_pct'].max()
            ]
        })
        
        risk_line = alt.Chart(risk_line_data).mark_rule(
            color='orange',
            size=2,
            strokeDash=[10, 5]
        ).encode(
            x='volatility_pct:Q'
        )

        # Combine all chart layers
        final_chart = (
            scatter + 
            frontier_line + 
            current_point + 
            risk_line
        ).properties(
            title='Efficient Frontier: Risk vs. Return Landscape'
        )
        
        st.altair_chart(final_chart, use_container_width=True)
        
        # Legend
        st.caption("""
        🔵 **Gray dots** = Random portfolios (all possible combinations)  
        🔴 **Red dashed line** = Efficient Frontier (optimal portfolios)  
        💎 **Green diamond** = Your current portfolio  
        🟠 **Orange line** = Your risk tolerance limit ({:.0f}%)
        """.format(params['target_risk']))

        # === YOUR POSITION ANALYSIS ===
        st.subheader("📍 Your Portfolio Position")
        
        col_pos1, col_pos2, col_pos3 = st.columns(3)
        
        with col_pos1:
            st.metric(
                "Your Risk",
                f"{current['volatility']*100:.2f}%",
                delta=f"{(current['volatility']*100 - params['target_risk']):.2f}% vs tolerance"
            )
        
        with col_pos2:
            st.metric(
                "Your Return",
                f"{current['return']*100:.2f}%",
                delta=None
            )
        
        with col_pos3:
            st.metric(
                "Your Sharpe Ratio",
                f"{current['sharpe']:.2f}",
                delta=None
            )

        # === OPTIMAL PORTFOLIO RECOMMENDATION ===
        st.divider()
        st.subheader("🎯 Optimal Portfolio Within Your Risk Tolerance")
        
        # ✅ FIXED: Proper if/else structure
        if optimal_at_risk is not None:
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                st.metric(
                    "Optimal Risk",
                    f"{optimal_at_risk['volatility']*100:.2f}%",
                    delta=f"{(optimal_at_risk['volatility']*100 - current['volatility']*100):.2f}% vs yours"
                )
            
            with col_opt2:
                st.metric(
                    "Optimal Return",
                    f"{optimal_at_risk['return']*100:.2f}%",
                    delta=f"{(optimal_at_risk['return']*100 - current['return']*100):.2f}% vs yours"
                )
            
            with col_opt3:
                st.metric(
                    "Optimal Sharpe",
                    f"{optimal_at_risk['sharpe']:.2f}",
                    delta=f"{(optimal_at_risk['sharpe'] - current['sharpe']):.2f} vs yours"
                )
            
            # Improvement message
            return_improvement = (optimal_at_risk['return'] - current['return']) * 100
            
            if return_improvement > 0.5:
                st.success(f"""
                ✅ **You can improve!** By optimizing your allocation, you could achieve 
                **{optimal_at_risk['return']*100:.2f}% return** (vs your current {current['return']*100:.2f}%) 
                while staying within your {params['target_risk']:.0f}% risk tolerance.
                
                That's an additional **{return_improvement:.2f}%** annual return for the same risk level!
                """)
            elif abs(return_improvement) <= 0.5:
                st.info(f"""
                👍 **Your portfolio is already near-optimal!** You're getting 
                **{current['return']*100:.2f}% return** at **{current['volatility']*100:.2f}% risk**, 
                which is very close to the efficient frontier.
                """)
            else:
                if current['volatility']*100 > params['target_risk']:
                    st.warning(f"""
                    ⚠️ **Your current portfolio ({current['volatility']*100:.2f}% risk) exceeds your stated risk tolerance ({params['target_risk']:.0f}%).**
                    
                    The optimal portfolio within your {params['target_risk']:.0f}% tolerance has:
                    - **{optimal_at_risk['return']*100:.2f}% return** (vs your {current['return']*100:.2f}%)
                    - **{optimal_at_risk['volatility']*100:.2f}% risk** (vs your {current['volatility']*100:.2f}%)
                    
                    **Options:**
                    1. **Increase risk tolerance** to {current['volatility']*100:.0f}%+ to accommodate your current strategy
                    2. **Reduce allocation** to match your {params['target_risk']:.0f}% tolerance (but accept lower return)
                    3. **Keep current allocation** if you're comfortable with the extra risk
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Your risk tolerance may be too restrictive.** The optimal portfolio within 
                    {params['target_risk']:.0f}% risk has lower return ({optimal_at_risk['return']*100:.2f}%) 
                    than your current portfolio ({current['return']*100:.2f}%).
                    
                    Consider increasing your risk tolerance or adjusting your allocation.
                    """)
            
            # === SHOW OPTIMAL WEIGHTS ===
            st.subheader("📊 Optimal Portfolio Allocation")
            st.write(f"These weights would give you **{optimal_at_risk['return']*100:.2f}% return** at **{optimal_at_risk['volatility']*100:.2f}% risk**:")

            # Get optimal weights (from optimizer — sums to 100% equity)
            optimal_weights = optimal_at_risk['weights']

            # ✅ V2: Scale by equity fraction so total = 100% with cash
            equity_scale = 1.0 - cash_pct
            optimal_weights_scaled = optimal_weights * equity_scale

            # Get company names
            ticker_names = get_stock_names(returns.columns.tolist())

            weights_comparison = pd.DataFrame({
                'Company': [ticker_names[t] for t in returns.columns.tolist()],
                'Ticker': returns.columns.tolist(),
                'Optimal Weight (%)': (optimal_weights_scaled * 100).round(2),
            })

            # Add cash row if applicable
            if cash_pct > 0:
                cash_row = pd.DataFrame([{
                    'Company': '💵 Cash (Risk-Free)',
                    'Ticker': 'CASH',
                    'Optimal Weight (%)': round(cash_pct * 100, 2)
                }])
                weights_comparison = pd.concat([weights_comparison, cash_row], ignore_index=True)

            # Sort equities by weight descending (keep cash at bottom)
            equity_rows = weights_comparison[weights_comparison['Ticker'] != 'CASH'].sort_values('Optimal Weight (%)', ascending=False)
            cash_rows = weights_comparison[weights_comparison['Ticker'] == 'CASH']

            # Total row
            total_val = round(weights_comparison['Optimal Weight (%)'].sum(), 2)
            total_row = pd.DataFrame([{'Company': 'TOTAL', 'Ticker': '—', 'Optimal Weight (%)': total_val}])
            weights_comparison = pd.concat([equity_rows, cash_rows, total_row], ignore_index=True)

            if cash_pct > 0:
                st.caption(f"✅ Equity weights scaled to {equity_scale*100:.1f}% of portfolio. Cash ({cash_pct*100:.1f}%) is fixed.")

            st.dataframe(
                weights_comparison,
                use_container_width=True,
                hide_index=True
            )

            # ✅ Apply button
            st.divider()
            if st.button("✅ Apply Optimal Weights to Portfolio", type="primary", use_container_width=True):
                # Store SCALED weights (sum to 1-cash_pct) so equity + cash = 100% ✅
                st.session_state["loaded_weights"] = optimal_weights_scaled.tolist()
                st.session_state["use_optimized"] = True
                st.session_state["opt_weights"] = optimal_weights_scaled.tolist()
                st.success("✅ Optimal weights loaded! Scroll up to see updated portfolio.")
                st.balloons()
                st.rerun()
        
        # ✅ ADDED: else block (only shows when optimal_at_risk is None)
        else:
            st.warning(f"""
            ⚠️ **No efficient portfolios found within your {params['target_risk']:.0f}% risk tolerance.**
            
            This means even the safest portfolio on the efficient frontier exceeds your risk limit.
            
            **Suggestions:**
            - Increase your risk tolerance slider
            - Reduce the minimum weight constraint (currently {lower_limit:.0f}%)
            - Consider adding lower-volatility assets to your portfolio
            """)
        
        # === TOP 5 EFFICIENT PORTFOLIOS ===
        st.divider()
        st.subheader("🏆 Top 5 Efficient Portfolios (Highest Sharpe Ratios)")
        
        st.write("""
        These are the best risk-adjusted portfolios on the efficient frontier. 
        They offer the highest returns per unit of risk taken.
        """)
        
        # Format the top 5 for display
        top5_display = top5.copy()
        top5_display['Return (%)'] = (top5_display['return'] * 100).round(2)
        top5_display['Risk (%)'] = (top5_display['volatility'] * 100).round(2)
        top5_display['Sharpe Ratio'] = top5_display['sharpe'].round(2)
        
        # Show only relevant columns
        top5_display = top5_display[['Return (%)', 'Risk (%)', 'Sharpe Ratio']].reset_index(drop=True)
        top5_display.index = top5_display.index + 1
        top5_display.index.name = 'Rank'
        
        st.dataframe(
            top5_display,
            use_container_width=True,
            hide_index=False
        )
        
        # Highlight if user's portfolio is in top 5
        user_sharpe = current['sharpe']
        top5_sharpes = top5['sharpe'].values
        
        if user_sharpe in top5_sharpes:
            rank = list(top5_sharpes).index(user_sharpe) + 1
            st.success(f"🌟 **Congratulations!** Your portfolio ranks #{rank} among the top efficient portfolios!")
        elif user_sharpe > top5_sharpes.min():
            st.info("👍 Your portfolio has a competitive Sharpe ratio, close to the top performers!")
        else:
            worst_top5_sharpe = top5_sharpes.min()
            st.warning(f"📊 Your Sharpe ratio ({user_sharpe:.2f}) could be improved. The 5th best portfolio has a Sharpe of {worst_top5_sharpe:.2f}.")
        
        # === EDUCATIONAL SECTION ===
        with st.expander("🎓 Understanding the Efficient Frontier"):
            st.markdown("""
            ### What am I looking at?
            
            **The Chart:**
            - Each gray dot = one possible portfolio allocation
            - X-axis = Risk (volatility)
            - Y-axis = Expected return
            - Color = Sharpe ratio (yellow = better risk-adjusted return)
            
            **The Red Line (Efficient Frontier):**
            - Shows **optimal portfolios** only
            - For each risk level, this is the **maximum possible return**
            - Portfolios below this line are "inefficient" (you can do better!)
            
            **Your Position:**
            - 💎 **On the line?** Perfect! You're optimized.
            - 💎 **Below the line?** You're leaving money on the table.
            - 💎 **Above the line?** Rare - check your data or you've found alpha!
            
            ### Why does this matter?
            
            **Diversification benefit:**
            - The frontier curves because of correlations between assets
            - You can reduce risk WITHOUT sacrificing return by smart allocation
            - That's the power of modern portfolio theory
            
            **Risk tolerance:**
            - The orange line shows your max acceptable risk
            - Find the highest point on the red line LEFT of the orange line
            - That's your optimal portfolio
            
            ### What should I do?
            
            1. **If below the frontier:** Use the optimizer to improve allocation
            2. **If on the frontier:** You're doing great!
            3. **If risk tolerance is too low:** Consider if you can accept slightly more risk for better returns
            4. **Check the Top 5 table:** See what allocations work best
            """)

    else:
        st.info("👆 Click '🚀 Generate Efficient Frontier' to visualize your portfolio's position in the risk-return landscape.")
# ============================================
# ============================================
# AI INSIGHTS DATA COLLECTION
# ============================================

# Collect all portfolio data for AI analysis
# ============================================
# AI INSIGHTS DATA COLLECTION
# ============================================

# Collect all portfolio data for AI analysis
try:
    # Get benchmark data if available
    alpha_value = None
    beta_value = None
    benchmark_name = None
    
    # Check if benchmark was calculated
    if 'benchmark_returns' in locals() and benchmark_returns is not None:
        aligned_df = pd.concat([portfolio_return, benchmark_returns], axis=1).dropna()
        aligned_df.columns = ["Portfolio", "Benchmark"]
        aligned_cumulative = (1 + aligned_df).cumprod()
        
        portfolio_total = aligned_cumulative.iloc[-1]["Portfolio"] - 1
        benchmark_total = aligned_cumulative.iloc[-1]["Benchmark"] - 1
        alpha_value = (portfolio_total - benchmark_total) * 100
        beta_value = portfolio_beta(aligned_df["Portfolio"], aligned_df["Benchmark"])
        benchmark_name = benchmark_names.get(benchmark_symbol, "Benchmark") if 'benchmark_symbol' in locals() else None
    
    # Get optimizer data if available
    current_sharpe_value = None
    optimized_sharpe_value = None
    
    if "opt_stats" in st.session_state:
        opt_ret, opt_vol, opt_sharpe = st.session_state["opt_stats"]
        raw_w = np.array(weights)
        eq_total = raw_w.sum()
        manual_weights_norm_ai = raw_w / eq_total if eq_total > 0 else raw_w
        mean_returns_opt = returns.mean() * 252
        cov_matrix_opt = returns.cov() * 252
        manual_ret, manual_vol, manual_sharpe = portfolio_stats(
            manual_weights_norm_ai, mean_returns_opt, cov_matrix_opt, risk_free
        )
        current_sharpe_value = manual_sharpe
        optimized_sharpe_value = opt_sharpe
    
    # Get frontier data if available
    frontier_position = None
    optimal_improvement_value = None
    target_risk_value = None  # ✅ NEW
    
    if 'frontier_data' in st.session_state:
        frontier_current = st.session_state['frontier_data']['current']
        frontier_optimal = st.session_state['frontier_data'].get('optimal_at_risk')
        frontier_params = st.session_state['frontier_data'].get('params', {})
        
        target_risk_value = frontier_params.get('target_risk')  # ✅ NEW
        
        if frontier_optimal is not None:
            current_return = frontier_current['return']
            optimal_return = frontier_optimal['return']
            
            if abs(current_return - optimal_return) <= 0.005:
                frontier_position = "on"
            elif current_return < optimal_return:
                frontier_position = "below"
            else:
                frontier_position = "above"
            
            optimal_improvement_value = (optimal_return - current_return) * 100
    
# Initialize Monte Carlo variables safely
    monte_p5 = monte_p50 = monte_p95 = monte_days = None

    if 'sim_results' in locals() and sim_results is not None:
        final_values = sim_results[-1, :]
        monte_p5 = np.percentile(final_values, 5)
        monte_p50 = np.percentile(final_values, 50)
        monte_p95 = np.percentile(final_values, 95)
        monte_days = num_days if 'num_days' in locals() else None
        
    # ── Risk Attribution (MCTR) data for AI ──────────────────────────────
    risk_attr_for_ai = None
    try:
        _mean_r_ai = returns.mean() * 252
        _cov_ai    = returns.cov() * 252
        _tnames_ai = get_stock_names(tickers_list)
        _mctr_ai   = compute_mctr(
            weights=np.array(weights),
            cov_matrix=_cov_ai,
            tickers=tickers_list,
            ticker_names=_tnames_ai,
            cash_pct=cash_pct
        )
        if not _mctr_ai.empty:
            _eq_ai  = _mctr_ai[_mctr_ai["Ticker"] != "CASH"]
            _w_eq   = np.array(weights)
            _w_norm = _w_eq / _w_eq.sum() if _w_eq.sum() > 0 else _w_eq
            _pvol   = np.sqrt(float(_w_norm @ _cov_ai.values @ _w_norm))
            _dr     = diversification_ratio(np.array(weights), _cov_ai)
            _top    = _eq_ai.loc[_eq_ai["% Risk Contribution"].idxmax()]
            # best diversifier: lowest risk-contrib / weight ratio
            _eq_nz  = _eq_ai[_eq_ai["Weight (%)"] > 0].copy()
            _eq_nz["_ratio"] = _eq_nz["% Risk Contribution"] / _eq_nz["Weight (%)"]
            _best   = _eq_nz.loc[_eq_nz["_ratio"].idxmin()]
            # compact per-asset breakdown string
            _breakdown = " | ".join(
                f"{r['Ticker']} {r['% Risk Contribution']:.1f}%"
                for _, r in _eq_ai.iterrows()
            )
            risk_attr_for_ai = {
                "diversification_ratio": round(_dr, 2),
                "equity_vol_pct":        round(_pvol * 100, 2),
                "top_risk_name":         _top["Company"],
                "top_risk_pct":          round(_top["% Risk Contribution"], 1),
                "top_risk_weight":       round(_top["Weight (%)"], 1),
                "best_diversifier_name": _best["Company"],
                "best_diversifier_risk_pct":   round(_best["% Risk Contribution"], 1),
                "best_diversifier_weight":     round(_best["Weight (%)"], 1),
                "risk_breakdown_str":    _breakdown,
            }
    except Exception:
        risk_attr_for_ai = None

    # ── Return Attribution (Brinson) data for AI ─────────────────────────
    return_attr_for_ai = None
    try:
        _bench_ret_ai, _ = benchmark_series(benchmark_symbol, adjusted_start, end_date)
        if hasattr(_bench_ret_ai.index, "tz") and _bench_ret_ai.index.tz is not None:
            _bench_ret_ai.index = _bench_ret_ai.index.tz_localize(None)
        _tnames_attr_ai = get_stock_names(tickers_list)
        _attr_ai = brinson_attribution(
            returns_data=returns,
            weights=np.array(weights),
            benchmark_returns=_bench_ret_ai,
            tickers=tickers_list,
            ticker_names=_tnames_attr_ai,
            cash_pct=cash_pct,
            risk_free_rate=risk_free,
        )
        _sdf    = _attr_ai["stock_df"]
        _eq_df  = _sdf[_sdf["Ticker"] != "CASH"]
        _boosters = _eq_df[_eq_df["Role"] == "🟢 Booster"]
        _drags    = _eq_df[_eq_df["Role"] == "🔴 Drag"]
        _top_b    = _eq_df.loc[_eq_df["Active Contribution (%)"].idxmax()] if not _eq_df.empty else None
        _top_d    = _eq_df.loc[_eq_df["Active Contribution (%)"].idxmin()] if not _eq_df.empty else None
        _cash_r   = _sdf[_sdf["Ticker"] == "CASH"]
        return_attr_for_ai = {
            "portfolio_return":  _attr_ai["portfolio_return"],
            "benchmark_return":  _attr_ai["benchmark_return"],
            "active_return":     _attr_ai["active_return_geo"],
            "num_boosters":      len(_boosters),
            "total_boost":       round(_boosters["Active Contribution (%)"].sum(), 2),
            "num_drags":         len(_drags),
            "total_drag":        round(_drags["Active Contribution (%)"].sum(), 2),
            "top_booster_name":  _top_b["Company"]    if _top_b is not None else "N/A",
            "top_booster_contrib": round(_top_b["Active Contribution (%)"], 2) if _top_b is not None else "N/A",
            "top_booster_return":  round(_top_b["Stock Return (%)"], 2)         if _top_b is not None else "N/A",
            "top_drag_name":     _top_d["Company"]    if _top_d is not None else "N/A",
            "top_drag_contrib":  round(_top_d["Active Contribution (%)"], 2)   if _top_d is not None else "N/A",
            "top_drag_return":   round(_top_d["Stock Return (%)"], 2)           if _top_d is not None else "N/A",
            "cash_pct":          cash_pct * 100,
            "cash_contrib":      round(_cash_r["Active Contribution (%)"].values[0], 2) if not _cash_r.empty else 0,
        }
    except Exception:
        return_attr_for_ai = None

    # Collect all data
    ai_portfolio_data = collect_portfolio_data(
        tickers_list=tickers_list,
        weights=weights,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        returns_data=returns,
        alpha=alpha_value,
        beta=beta_value,
        benchmark_label=benchmark_name,
        current_sharpe=current_sharpe_value,
        optimized_sharpe=optimized_sharpe_value,
        frontier_position=frontier_position,
        optimal_improvement=optimal_improvement_value,
        target_risk=target_risk_value,
        monte_carlo_p5=monte_p5,
        monte_carlo_p50=monte_p50,
        monte_carlo_p95=monte_p95,
        monte_carlo_days=monte_days,
        cash_pct=cash_pct * 100,  # ✅ V2: Cash component (pass as % for AI display)
        risk_attribution_data=risk_attr_for_ai,
        return_attribution_data=return_attr_for_ai,
    )
    
except Exception as e:
    st.error(f"Error collecting AI data: {str(e)}")
    ai_portfolio_data = None


if 'metrics' in locals() and ai_portfolio_data is not None:
    render_sidebar_summary(ai_portfolio_data)

# ============================================
# MCTR / RISK ATTRIBUTION TAB
# ============================================
with tab_mctr:
    st.header("📐 Risk Attribution (MCTR)")
    st.write("""
    **Marginal Contribution to Risk (MCTR)** decomposes your portfolio's total volatility
    into each asset's individual contribution. This reveals which positions are truly
    driving your risk — and which are providing diversification benefit.
    """)

    # Compute annualised cov matrix from equity returns only
    mean_returns_mctr = returns.mean() * 252
    cov_matrix_mctr = returns.cov() * 252

    # Get company names
    ticker_names_mctr = get_stock_names(tickers_list)

    # Compute MCTR — pass the actual weights (already scaled by 1-cash_pct)
    mctr_df = compute_mctr(
        weights=np.array(weights),
        cov_matrix=cov_matrix_mctr,
        tickers=tickers_list,
        ticker_names=ticker_names_mctr,
        cash_pct=cash_pct
    )

    if mctr_df.empty:
        st.warning("⚠️ Unable to compute risk attribution. Check your portfolio inputs.")
    else:
        # ── Portfolio-level stats ──────────────────────────────────────────
        w_eq = np.array(weights)
        w_eq_norm = w_eq / w_eq.sum() if w_eq.sum() > 0 else w_eq
        port_vol_mctr = np.sqrt(float(w_eq_norm @ cov_matrix_mctr.values @ w_eq_norm))
        dr = diversification_ratio(np.array(weights), cov_matrix_mctr)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Portfolio Volatility (Equity)", f"{port_vol_mctr*100:.2f}%")
        col_m1.caption("Annualised vol of equity portion only.")

        col_m2.metric("Diversification Ratio", f"{dr:.2f}x")
        col_m2.caption("Weighted avg vol ÷ portfolio vol. Higher = better diversified.")

        # Largest risk contributor
        equity_only = mctr_df[mctr_df["Ticker"] != "CASH"]
        if not equity_only.empty:
            top_risk_asset = equity_only.loc[equity_only["% Risk Contribution"].idxmax()]
            col_m3.metric(
                "Largest Risk Contributor",
                top_risk_asset["Company"][:20],
                f"{top_risk_asset['% Risk Contribution']:.1f}% of risk"
            )
            col_m3.caption(f"Ticker: {top_risk_asset['Ticker']}")

        st.divider()

        # ── % Risk Contribution bar chart ─────────────────────────────────
        st.subheader("📊 % Risk Contribution per Asset")
        st.caption("How much of the portfolio's total volatility each position contributes.")

        chart_df = mctr_df[mctr_df["Ticker"] != "CASH"].copy()
        chart_df = chart_df.sort_values("% Risk Contribution", ascending=False)

        bar_chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Company:N",
                    sort="-y",
                    title=None,
                    axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelLimit=150)
                ),
                y=alt.Y("% Risk Contribution:Q", title="% of Total Portfolio Risk"),
                color=alt.Color(
                    "% Risk Contribution:Q",
                    scale=alt.Scale(scheme="reds"),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("Company:N", title="Asset"),
                    alt.Tooltip("Ticker:N", title="Ticker"),
                    alt.Tooltip("Weight (%):Q", title="Weight (%)", format=".2f"),
                    alt.Tooltip("% Risk Contribution:Q", title="% Risk Contribution", format=".2f"),
                    alt.Tooltip("MCTR (%):Q", title="MCTR (%)", format=".4f"),
                ]
            )
            .properties(height=380)
        )
        st.altair_chart(bar_chart, use_container_width=True)

        # ── Full attribution table ─────────────────────────────────────────
        st.subheader("📋 Full Risk Attribution Table")

        # Sort equities by risk contribution descending, cash at bottom
        equity_rows_mctr = mctr_df[mctr_df["Ticker"] != "CASH"].sort_values(
            "% Risk Contribution", ascending=False
        )
        cash_rows_mctr = mctr_df[mctr_df["Ticker"] == "CASH"]

        # Total row
        total_mctr_row = pd.DataFrame([{
            "Company": "TOTAL",
            "Ticker": "—",
            "Weight (%)": round(mctr_df["Weight (%)"].sum(), 2),
            "MCTR (%)": "",
            "% Risk Contribution": round(equity_only["% Risk Contribution"].sum(), 2),
            "Abs Risk Contrib (%)": "",
        }])

        display_df = pd.concat(
            [equity_rows_mctr, cash_rows_mctr, total_mctr_row],
            ignore_index=True
        )

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        # ── Concentration warning ──────────────────────────────────────────
        if not equity_only.empty:
            top_pct = equity_only["% Risk Contribution"].max()
            if top_pct > 40:
                st.warning(
                    f"⚠️ **Concentration Alert:** {top_risk_asset['Company']} contributes "
                    f"**{top_pct:.1f}%** of total portfolio risk. Consider reducing this position."
                )
            elif top_pct > 25:
                st.info(
                    f"ℹ️ **Moderate Concentration:** {top_risk_asset['Company']} drives "
                    f"**{top_pct:.1f}%** of risk. Monitor this exposure."
                )
            else:
                st.success("✅ **Well Diversified:** No single asset dominates portfolio risk.")

        # ── Explainer ─────────────────────────────────────────────────────
        with st.expander("📖 How to read this table"):
            st.markdown("""
**Weight (%)** — How much of your total portfolio is in this asset.

**MCTR (%)** — *Marginal Contribution to Risk.* How much portfolio volatility increases
if you add 1% more to this position. Higher MCTR = this asset is more "risk-dense."

**% Risk Contribution** — This asset's share of total portfolio risk.
All equity assets sum to 100%. Cash always contributes 0%.

**Abs Risk Contrib (%)** — MCTR × Weight. The raw annualised risk each asset adds.
These sum to total portfolio volatility.

**Diversification Ratio** — If all assets were perfectly correlated, DR = 1.0.
A DR of 1.3x means diversification is cutting your risk by ~23% vs holding each asset alone.

**Key insight:** An asset with a large *weight* but low *% risk contribution* is
acting as a diversifier. An asset with a small weight but large risk contribution
is punching above its weight in terms of risk — consider trimming it.
            """)

# ============================================
# ACTIVE RETURN ATTRIBUTION TAB
# ============================================
# ============================================
# ACTIVE RETURN ATTRIBUTION TAB
# ============================================
with tab_return_attr:
    st.header("📊 Active Return Attribution")
    st.write(f"""
    Shows how much each position **contributed to or detracted from** your portfolio's
    active return vs **{benchmark_label}**. Boosters beat the benchmark; drags lagged it.
    """)

    try:
        bench_ret_attr, _ = benchmark_series(benchmark_symbol, adjusted_start, end_date)

        if hasattr(bench_ret_attr.index, "tz") and bench_ret_attr.index.tz is not None:
            bench_ret_attr.index = bench_ret_attr.index.tz_localize(None)
        bench_ret_attr.name = "bench"

        # Align portfolio and benchmark (geometric display)
        port_bench_aligned = pd.concat(
            [portfolio_return.rename("Portfolio"), bench_ret_attr.rename("Benchmark")],
            axis=1
        ).dropna()

        if port_bench_aligned.empty or len(port_bench_aligned) < 5:
            st.warning("⚠️ Not enough overlapping data between portfolio and benchmark.")
        else:
            display_port_return  = float((1 + port_bench_aligned["Portfolio"]).prod() - 1) * 100
            display_bench_return = float((1 + port_bench_aligned["Benchmark"]).prod() - 1) * 100

            ticker_names_attr = get_stock_names(tickers_list)

            attr_result = brinson_attribution(
                returns_data=returns,
                weights=np.array(weights),
                benchmark_returns=bench_ret_attr,
                tickers=tickers_list,
                ticker_names=ticker_names_attr,
                cash_pct=cash_pct,
                risk_free_rate=risk_free,
            )

            stock_df = attr_result["stock_df"].copy()
            active_return_geo = attr_result["active_return_geo"]

            # ── Performance summary
            st.subheader("📈 Performance Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Portfolio Return", f"{display_port_return:.2f}%")
            c2.metric(f"{benchmark_label} Return", f"{display_bench_return:.2f}%")
            c3.metric("Active Return", f"{active_return_geo:.2f}%")

            st.caption("⚖️ Contributions are adjusted for the **compounding effect** so their sum exactly matches the Portfolio's Geometric Active Return.")

            st.divider()

            # ── Boosters vs Drags (+ Cash if applicable)
            equity_df = stock_df[stock_df["Ticker"] != "CASH"].copy()
            boosters = equity_df[equity_df["Role"] == "🟢 Booster"].sort_values(
                "Active Contribution (%)", ascending=False
            )
            drags = equity_df[equity_df["Role"] == "🔴 Drag"].sort_values(
                "Active Contribution (%)"
            )
            cash_row_attr = stock_df[stock_df["Ticker"] == "CASH"]

            # 3 columns when cash is present, 2 otherwise
            if cash_pct > 0 and not cash_row_attr.empty:
                col_b, col_d, col_c = st.columns(3)
            else:
                col_b, col_d = st.columns(2)
                col_c = None

            with col_b:
                st.subheader(f"🟢 Boosters ({len(boosters)})")
                st.metric("Total Boost", f"+{boosters['Active Contribution (%)'].sum():.2f}%")
                st.dataframe(
                    boosters[["Company", "Portfolio Wt (%)", "Stock Return (%)", "Excess Return (%)", "Active Contribution (%)"]],
                    hide_index=True, use_container_width=True
                )

            with col_d:
                st.subheader(f"🔴 Drags ({len(drags)})")
                st.metric("Total Drag", f"{drags['Active Contribution (%)'].sum():.2f}%")
                st.dataframe(
                    drags[["Company", "Portfolio Wt (%)", "Stock Return (%)", "Excess Return (%)", "Active Contribution (%)"]],
                    hide_index=True, use_container_width=True
                )

            if col_c is not None:
                with col_c:
                    _cash_contrib = cash_row_attr["Active Contribution (%)"].values[0]
                    _cash_ret     = cash_row_attr["Stock Return (%)"].values[0]
                    _cash_role    = cash_row_attr["Role"].values[0]
                    _cash_sign    = "+" if _cash_contrib >= 0 else ""
                    st.subheader("💵 Cash")
                    st.metric(
                        f"Cash Contribution ({cash_pct*100:.0f}%)",
                        f"{_cash_sign}{_cash_contrib:.2f}%",
                        help=f"Cash earned {_cash_ret:.2f}% (risk-free rate). Role: {_cash_role}"
                    )
                    st.caption(
                        f"Earned **{_cash_ret:.2f}%** vs benchmark **{display_bench_return:.2f}%**. "
                        f"Cash is a drag when the market rises faster than the risk-free rate."
                    )

            st.divider()

            # ── Full table with reconciliation check
            total_contrib_sum = round(stock_df["Active Contribution (%)"].sum(), 4)
            st.caption(f"Check: Boost + Drag = {total_contrib_sum:.2f}% | Active Return = {active_return_geo:.2f}%")

            st.dataframe(stock_df, hide_index=True, use_container_width=True)


            # ── Active contribution bar chart
            st.subheader("📊 Active Contribution per Stock")
            st.caption(f"How much each position added (+) or subtracted (−) from your return vs {benchmark_label}.")

            chart_df_attr = equity_df.sort_values("Active Contribution (%)", ascending=False).copy()
            chart_df_attr["Color"] = chart_df_attr["Active Contribution (%)"].apply(
                lambda x: "positive" if x >= 0 else "negative"
            )

            contrib_chart = (
                alt.Chart(chart_df_attr)
                .mark_bar()
                .encode(
                    x=alt.X("Company:N", sort="-y", title=None,
                            axis=alt.Axis(labelAngle=-45, labelLimit=150, labelOverlap=False)),
                    y=alt.Y("Active Contribution (%):Q", title="Active Contribution (%)"),
                    color=alt.Color(
                        "Color:N",
                        scale=alt.Scale(domain=["positive", "negative"], range=["#4ade80", "#ef4444"]),
                        legend=None
                    ),
                    tooltip=[
                        alt.Tooltip("Company:N", title="Stock"),
                        alt.Tooltip("Portfolio Wt (%):Q", title="Weight (%)", format=".2f"),
                        alt.Tooltip("Stock Return (%):Q", title="Stock Return (%)", format=".2f"),
                        alt.Tooltip("Excess Return (%):Q", title="Excess vs Benchmark (%)", format=".2f"),
                        alt.Tooltip("Active Contribution (%):Q", title="Active Contribution (%)", format=".4f"),
                    ]
                )
                .properties(height=360)
            )
            zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                color="white", strokeDash=[4, 4], opacity=0.4
            ).encode(y="y:Q")
            st.altair_chart(contrib_chart + zero_line, use_container_width=True)

            # ── Full table
            st.subheader("📋 Full Attribution Table")

            sorted_equity = equity_df.sort_values("Active Contribution (%)", ascending=False)
            cash_rows_attr = stock_df[stock_df["Ticker"] == "CASH"]

            total_attr_row = pd.DataFrame([{
                "Company":                 "TOTAL",
                "Ticker":                  "—",
                "Portfolio Wt (%)":        round(stock_df["Portfolio Wt (%)"].sum(), 2),
                "Stock Return (%)":        "—",
                "Benchmark Ret (%)":       round(display_bench_return, 2),
                "Excess Return (%)":       "—",
                "Active Contribution (%)": total_contrib_sum,
                "Role":                    "—",
            }])

            display_attr_df = pd.concat(
                [sorted_equity, cash_rows_attr, total_attr_row],
                ignore_index=True
            )
            st.dataframe(display_attr_df, hide_index=True, use_container_width=True)

            if cash_pct > 0 and display_bench_return > 0:
                cash_rows_disp = stock_df[stock_df["Ticker"] == "CASH"]
                if not cash_rows_disp.empty:
                    cash_drag_val = cash_rows_disp["Active Contribution (%)"].values[0]
                    cash_ret_val = cash_rows_disp["Stock Return (%)"].values[0]
                    st.caption(
                        f"💵 Cash ({cash_pct*100:.1f}%) earned **{cash_ret_val:.2f}%** "
                        f"(risk-free rate {risk_free*100:.2f}% p.a. compounded) and contributed "
                        f"**{cash_drag_val:.2f}%** to active return vs {benchmark_label}."
                    )

            with st.expander("📖 Methodology & how to read this"):
                st.markdown(f"""
    **Performance Summary** uses true geometric (compound) returns.
    - Portfolio Return: {display_port_return:.2f}% | {benchmark_label}: {display_bench_return:.2f}% | Active: {active_return_geo:.2f}%

    **Active Contribution per Stock**
    `Base Contribution = Portfolio Weight × (Stock Geo Return − Benchmark Geo Return)`

    Because of volatility drag and daily rebalancing, the sum of base contributions will slightly miss the true geometric portfolio return. We calculate this **Compounding Residual** and distribute it across all assets proportional to their weight. 

    This ensures that **all individual contributions sum exactly** to the TOTAL Active Return, giving you a mathematically reconciled view of performance.

    **Boosters** — beat the benchmark → positive contribution.
    **Drags** — lagged the benchmark → negative contribution.
    **Cash** earns the risk-free rate. It acts as a drag when the benchmark is rising faster than the risk-free rate.
                """)

    except Exception as e:
        st.error(f"Error computing return attribution: {str(e)}")
        st.info("Ensure portfolio data is loaded and the date range has sufficient history.")
# ============================================
# AI ANALYSIS TAB
# ============================================
with tab_ai:
    if ai_portfolio_data is not None:
        render_full_analysis_tab(ai_portfolio_data)
    else:
        st.warning("⚠️ Unable to generate AI analysis. Please ensure portfolio data is loaded.")
        st.info("💡 Make sure you have:\n- Selected tickers\n- Set weights (totaling 100%)\n- Portfolio data downloaded successfully")

# ============================================
# SIDEBAR: SAVE/LOAD PORTFOLIOS
# ============================================
st.sidebar.markdown("---")

# ✅ LOAD PORTFOLIO SECTION (with Rename/Delete)
try:
    all_portfolios = portfolios_ws.get_all_records()
    
    # Filter: Show only ACTIVE portfolios for current user
    user_portfolios = [
        p for p in all_portfolios 
        if st.session_state.get("user_email") and 
           p.get("email", "").lower() == st.session_state["user_email"].lower() and
           p.get("status", "active").lower() == "active"
    ]

    if user_portfolios:
        st.sidebar.subheader("📂 Saved Portfolios")
        
        portfolio_options = [f"{p['portfolio_name']}" for p in user_portfolios]
        
        selected_display = st.sidebar.selectbox(
            "Choose a portfolio:",
            portfolio_options
        )
        
        selected_index = portfolio_options.index(selected_display)
        selected_portfolio = user_portfolios[selected_index]
        
        # Action buttons
        col_load, col_rename, col_delete = st.sidebar.columns(3)
        
        with col_load:
            if st.button("🚀 Load", use_container_width=True):
                loaded_tickers = selected_portfolio["tickers"].split(",")
                loaded_weights = [float(w) for w in selected_portfolio["weight"].split(",")]
                
                st.session_state["loaded_tickers"] = loaded_tickers
                st.session_state["loaded_weights"] = loaded_weights

                # ✅ V2: Restore cash component if saved
                saved_cash = selected_portfolio.get("cash_pct", 0)
                try:
                    st.session_state["cash_weight_pct"] = float(saved_cash) if saved_cash != "" else 0.0
                except (ValueError, TypeError):
                    st.session_state["cash_weight_pct"] = 0.0
                
                if selected_portfolio.get("optimized_weights"):
                    opt_weights_str = selected_portfolio["optimized_weights"]
                    if opt_weights_str:
                        st.session_state["opt_weights"] = [float(w) for w in opt_weights_str.split(",")]
                        st.session_state["use_optimized"] = True
                
                st.sidebar.success(f"✅ Loading {selected_portfolio['portfolio_name']}...")
                st.rerun()
        
        with col_rename:
            if st.button("✏️ Edit", use_container_width=True):
                st.session_state["renaming_portfolio"] = selected_portfolio["portfolio_id"]
                st.rerun()
        
        with col_delete:
            if st.button("Delete", use_container_width=True):
                st.session_state["deleting_portfolio"] = selected_portfolio["portfolio_id"]
                st.rerun()
        
        # Rename dialog
        if st.session_state.get("renaming_portfolio") == selected_portfolio["portfolio_id"]:
            new_name = st.sidebar.text_input(
                "New portfolio name:",
                value=selected_portfolio["portfolio_name"],
                key="rename_input"
            )
            
            col_confirm, col_cancel = st.sidebar.columns(2)
            
            with col_confirm:
                if st.button("✅ Confirm", use_container_width=True):
                    if new_name and new_name.strip():
                        success = rename_portfolio(
                            selected_portfolio["portfolio_id"],
                            new_name.strip(),
                            portfolios_ws
                        )
                        if success:
                            st.sidebar.success("✅ Renamed!")
                            del st.session_state["renaming_portfolio"]
                            st.rerun()
                    else:
                        st.sidebar.error("Name cannot be empty")
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    del st.session_state["renaming_portfolio"]
                    st.rerun()
        
        # Delete confirmation dialog
        if st.session_state.get("deleting_portfolio") == selected_portfolio["portfolio_id"]:
            st.sidebar.warning(f"⚠️ Delete {selected_portfolio['portfolio_name']}?")
            
            col_confirm, col_cancel = st.sidebar.columns(2)
            
            with col_confirm:
                if st.button("✅ Delete", use_container_width=True):
                    success = soft_delete_portfolio(
                        selected_portfolio["portfolio_id"],
                        portfolios_ws
                    )
                    if success:
                        st.sidebar.success("✅ Portfolio deleted!")
                        del st.session_state["deleting_portfolio"]
                        st.rerun()
            
            with col_cancel:
                if st.button("❌ Cancel", use_container_width=True):
                    del st.session_state["deleting_portfolio"]
                    st.rerun()

except Exception as e:
    st.sidebar.error(f"Error loading portfolios: {e}")


# ============================================
# PDF EXPORT
# ============================================
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Export PDF Report")

if st.sidebar.button("📥 Download PDF Report", use_container_width=True, type="primary"):
    with st.sidebar:
        with st.spinner("Building PDF…"):
            try:
                # ── Portfolio basics ──────────────────────────────────────
                _ticker_names_list = [ticker_names.get(t, t) for t in tickers_list]
                _weights_pct = [w * 100 for w in weights]

                # Drawdown dates (computed in risk tab, replicate here)
                try:
                    _rm = cumulative.cummax()
                    _dd = (cumulative - _rm) / _rm
                    _dd_trough = _dd.idxmin()
                    _dd_peak   = cumulative.loc[:_dd_trough].idxmax()
                    dd_peak_str   = _dd_peak.strftime("%d %b %Y")
                    dd_trough_str = _dd_trough.strftime("%d %b %Y")
                except Exception:
                    dd_peak_str = dd_trough_str = None

                pdf_data = {
                    # Always-present ──────────────────────────────────────
                    "tickers_list":       tickers_list,
                    "ticker_names_list":  _ticker_names_list,
                    "weights_pct":        _weights_pct,
                    "cash_pct":           cash_pct,
                    "date_range":         f"{start_date.strftime('%d %b %Y')} – {end_date.strftime('%d %b %Y')}",
                    "risk_free":          risk_free,
                    "num_assets":         len(tickers_list),
                    "metrics":            metrics,
                    "dd_peak":            dd_peak_str,
                    "dd_trough":          dd_trough_str,

                    # Conditional sections – populated below if available ──
                    "benchmark":    None,
                    "monte_carlo":  None,
                    "optimizer":    None,
                    "frontier":     None,
                    "mctr":         None,
                    "return_attr":  None,
                    "ai_analysis":  None,
                }

                # ── Benchmark ────────────────────────────────────────────
                try:
                    if "benchmark_returns" in dir() or "benchmark_returns" in locals() \
                            or ("benchmark_symbol" in dir()):
                        _bench_r, _ = benchmark_series(benchmark_symbol, adjusted_start, end_date)
                        _aligned = pd.concat(
                            [portfolio_return, _bench_r], axis=1
                        ).dropna()
                        _aligned.columns = ["Portfolio", "Bench"]
                        _acum = (1 + _aligned).cumprod()
                        _pt = _acum.iloc[-1]["Portfolio"] - 1
                        _bt = _acum.iloc[-1]["Bench"] - 1
                        _beta = portfolio_beta(_aligned["Portfolio"], _aligned["Bench"])
                        pdf_data["benchmark"] = {
                            "label":       benchmark_names.get(benchmark_symbol, "Benchmark"),
                            "port_total":  _pt,
                            "bench_total": _bt,
                            "alpha":       (_pt - _bt) * 100,
                            "beta":        _beta,
                        }
                except Exception:
                    pass

                # ── Monte Carlo ───────────────────────────────────────────
                try:
                    if "sim_results" in locals() and sim_results is not None:
                        _fv = sim_results[-1, :]
                        pdf_data["monte_carlo"] = {
                            "p5":   float(np.percentile(_fv, 5)),
                            "p50":  float(np.percentile(_fv, 50)),
                            "p95":  float(np.percentile(_fv, 95)),
                            "days": int(num_days) if "num_days" in locals() else "N/A",
                        }
                except Exception:
                    pass

                # ── Optimizer ─────────────────────────────────────────────
                try:
                    if "opt_stats" in st.session_state and "opt_weights" in st.session_state:
                        _opt_r, _opt_v, _opt_s = st.session_state["opt_stats"]
                        _opt_w_raw = np.array(st.session_state.get(
                            "opt_weights_raw", st.session_state["opt_weights"]
                        ))
                        if abs(_opt_w_raw.sum() - 1.0) > 0.01:
                            _opt_w_raw = _opt_w_raw / _opt_w_raw.sum()
                        _mr = returns.mean() * 252
                        _mc = returns.cov() * 252
                        _raw_m = np.array(weights)
                        _eq_t = _raw_m.sum()
                        _man_n = _raw_m / _eq_t if _eq_t > 0 else _raw_m
                        _man_r, _man_v, _man_s = portfolio_stats(_man_n, _mr, _mc, risk_free)
                        _opt_w_display = np.array(st.session_state["opt_weights"])
                        pdf_data["optimizer"] = {
                            "manual_ret":    _man_r,
                            "manual_vol":    _man_v,
                            "manual_sharpe": _man_s,
                            "opt_ret":       _opt_r,
                            "opt_vol":       _opt_v,
                            "opt_sharpe":    _opt_s,
                            "opt_weights_pct": [w * 100 for w in _opt_w_display],
                        }
                except Exception:
                    pass

                # ── Efficient Frontier ────────────────────────────────────
                try:
                    if "frontier_data" in st.session_state:
                        _fd = st.session_state["frontier_data"]
                        _cur = _fd["current"]
                        _top5 = _fd.get("top5")
                        _pos = "on"
                        if _fd.get("optimal_at_risk") is not None:
                            _gap = _fd["optimal_at_risk"]["return"] - _cur["return"]
                            _pos = "below" if _gap > 0.005 else ("above" if _gap < -0.005 else "on")
                        pdf_data["frontier"] = {
                            "current":  _cur,
                            "position": _pos,
                            "top5":     _top5,
                        }
                except Exception:
                    pass

                # ── MCTR / Risk Attribution ───────────────────────────────
                try:
                    _cov_pdf  = returns.cov() * 252
                    _tn_pdf   = get_stock_names(tickers_list)
                    _mctr_pdf = compute_mctr(
                        weights=np.array(weights),
                        cov_matrix=_cov_pdf,
                        tickers=tickers_list,
                        ticker_names=_tn_pdf,
                        cash_pct=cash_pct,
                    )
                    if not _mctr_pdf.empty:
                        _weq = np.array(weights)
                        _wn  = _weq / _weq.sum() if _weq.sum() > 0 else _weq
                        _ev  = float(np.sqrt(_wn @ _cov_pdf.values @ _wn)) * 100
                        _dr  = diversification_ratio(np.array(weights), _cov_pdf)
                        pdf_data["mctr"] = {
                            "df":         _mctr_pdf,
                            "equity_vol": _ev,
                            "div_ratio":  _dr,
                        }
                except Exception:
                    pass

                # ── Return Attribution ────────────────────────────────────
                try:
                    _bench_ret_pdf, _ = benchmark_series(
                        benchmark_symbol, adjusted_start, end_date
                    )
                    if hasattr(_bench_ret_pdf.index, "tz") and _bench_ret_pdf.index.tz:
                        _bench_ret_pdf.index = _bench_ret_pdf.index.tz_localize(None)
                    _tn_ra = get_stock_names(tickers_list)
                    _attr_pdf = brinson_attribution(
                        returns_data=returns,
                        weights=np.array(weights),
                        benchmark_returns=_bench_ret_pdf,
                        tickers=tickers_list,
                        ticker_names=_tn_ra,
                        cash_pct=cash_pct,
                        risk_free_rate=risk_free,
                    )
                    _sdf = _attr_pdf["stock_df"]
                    _eq_ra = _sdf[_sdf["Ticker"] != "CASH"]
                    _cash_ra = _sdf[_sdf["Ticker"] == "CASH"]
                    # Build display df: equities → cash → total
                    _total_ra = pd.DataFrame([{
                        "Company": "TOTAL", "Ticker": "—",
                        "Portfolio Wt (%)": round(_sdf["Portfolio Wt (%)"].sum(), 2),
                        "Stock Return (%)": "—",
                        "Excess Return (%)": "—",
                        "Active Contribution (%)": round(
                            _sdf["Active Contribution (%)"].sum(), 4),
                    }])
                    _display_ra = pd.concat(
                        [_eq_ra.sort_values("Active Contribution (%)", ascending=False),
                         _cash_ra, _total_ra],
                        ignore_index=True
                    )
                    pdf_data["return_attr"] = {
                        "df":               _display_ra,
                        "portfolio_return": _attr_pdf["portfolio_return"],
                        "benchmark_return": _attr_pdf["benchmark_return"],
                        "active_return":    _attr_pdf["active_return_geo"],
                        "num_boosters":     len(_eq_ra[_eq_ra["Role"] == "🟢 Booster"]),
                        "num_drags":        len(_eq_ra[_eq_ra["Role"] == "🔴 Drag"]),
                    }
                except Exception:
                    pass

                # ── AI Analysis ───────────────────────────────────────────
                if st.session_state.get("ai_analysis_generated") and \
                        st.session_state.get("ai_full_analysis"):
                    pdf_data["ai_analysis"] = st.session_state["ai_full_analysis"]

                # ── Generate PDF ──────────────────────────────────────────
                pdf_bytes = generate_portfolio_pdf(pdf_data)

                st.sidebar.download_button(
                    label="⬇️ Save PDF",
                    data=pdf_bytes,
                    file_name=f"portfolio_report_{datetime.date.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as e:
                st.sidebar.error(f"PDF generation failed: {e}")

# ✅ SAVE PORTFOLIO SECTION (UNCHANGED - Still here!)
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Save Current Portfolio")

portfolio_name = st.sidebar.text_input("Portfolio Name", placeholder="e.g., Tech Growth")

if st.sidebar.button("💾 Save Portfolio Snapshot"):
    try:
        # Get user's name from users_df
        user_row = users_df[users_df["email"] == st.session_state["user_email"].lower()]
        user_name = user_row.iloc[0]["name"] if not user_row.empty else "Unknown"

        save_portfolio(
            st.session_state["user_email"],
            user_name,
            portfolio_name if portfolio_name else "Untitled Portfolio",
            tickers_list,
            weights,
            portfolios_ws,
            st.session_state.get("opt_weights") if st.session_state["use_optimized"] else None,
            cash_pct=cash_weight_pct  # ✅ V2: Save cash component
        )
        st.sidebar.success("✅ Saved!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")


st.sidebar.markdown("---")
if st.session_state.get("user_email"):
    st.sidebar.caption(f"👤 {st.session_state['user_email']}")

if st.sidebar.button("🚪 Logout", use_container_width=True, type="secondary"):
    logout(cookies)

# ============================================
# FOOTER & DISCLAIMER (Shifted Up)
# ============================================

# 1. The "Harbhajan The Great" Footer (Fixed Higher)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 90px;  /* ⬆️ Increased from 70px */
        right: 15px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .3px;
        color: #b00020;
        border: 1.5px solid #b00020;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(0,0,0,0.2);
        backdrop-filter: blur(2px);
        z-index: 1000;
        pointer-events: none;
    }
    </style>
    <div class="footer">HSTG@FT</div>
    """,
    unsafe_allow_html=True
)

# 2. The Disclaimer Button (Shifted Higher)
# 2. The Disclaimer Button (Shifted Higher)
st.markdown(
    """
    <style>
    .disclaimer-container {
        position: fixed;
        bottom: 50px;
        right: 15px;
        z-index: 1001;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }

    .disclaimer-btn {
        background-color: #b00020;
        color: white;
        border: none;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        list-style: none;
        transition: background 0.2s;
    }
    .disclaimer-btn:hover {
        background-color: #d9002a;
    }
    .disclaimer-btn::-webkit-details-marker {
        display: none;
    }

    .disclaimer-content {
        position: absolute;
        bottom: 45px;
        right: 0;
        width: 340px;
        background-color: #1a1a1a;
        color: #cccccc;
        padding: 12px;
        border: 1px solid #333;
        border-left: 3px solid #b00020;
        border-radius: 8px;
        font-size: 11px;
        line-height: 1.4;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        animation: fadeIn 0.2s ease-out;
        text-align: left;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .disc-header {
        font-weight: 700;
        color: #fff;
        border-bottom: 1px solid #444;
        padding-bottom: 4px;
        margin-bottom: 8px;
        display: block;
    }
    </style>

    <div class="disclaimer-container">
    <details>
    <summary class="disclaimer-btn">⚠️ Disclaimer & Methodology</summary>
    <div class="disclaimer-content">
    <span class="disc-header">DISCLAIMER & METHODOLOGY</span>
    <strong>1. Methodology (Lump‑Sum Assumption)</strong><br>
    This dashboard assumes your portfolio was <u>fully invested</u> starting from the selected Start Date.
    It does <strong>not</strong> account for:
    <ul style="margin: 4px 0 8px 15px; padding: 0; color: #b0b0b0;">
    <li>Gradual investments (SIP/DCA)</li>
    <li>Mid‑period rebalancing</li>
    <li>Taxes, transaction fees, or dividends (unless using total return data)</li>
    </ul>
    <em>Implication: Metrics are an approximation of historical performance, not a precise reflection of your actual investment journey.</em>
    <br><br>
    <strong>2. Data & Accuracy</strong><br>
    Market data is sourced from third‑party providers and may be delayed or contain errors. We do not guarantee the accuracy of this data.
    <br><br>
    <strong>3. Educational Use Only</strong><br>
    All charts, metrics and AI insights are for educational purposes only. They should not be interpreted as financial advice or guarantees of future outcomes.
    </div>
    </details>
    </div>
    """,
    unsafe_allow_html=True
)