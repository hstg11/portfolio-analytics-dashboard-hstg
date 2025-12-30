import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

from auth import *
from portfolio import *
from optimizer import *
from config import *

st.set_page_config(layout="wide")

# ============================================
# INITIALIZE SERVICES
# ============================================
cookies = init_cookie_manager()
users_ws, portfolios_ws = init_google_sheets()
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

# ============================================
# SIGNUP GATE (BEFORE ANYTHING ELSE)
# ============================================
if not st.session_state["signed_up"]:
    show_signup_gate(users_df, users_ws, cookies)

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("📊 Portfolio Performance & Risk Dashboard")
st.write("Analyze returns, risk, and benchmark performance in one place.")

st.sidebar.header("Portfolio Inputs")

# Ticker input - reads from loaded portfolio
if st.session_state["loaded_tickers"] is not None:
    default_tickers = ",".join(st.session_state["loaded_tickers"])
    st.session_state["loaded_tickers"] = None
else:
    default_tickers = "AAPL,MSFT,AMZN,META,GOOG"

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    value=default_tickers
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

tickers_list = [t.strip() for t in tickers.split(",")]

st.sidebar.info("""Input Guide for Indian Stocks - NSE stocks → (Ticker + `.NS`) , e.g. `RELIANCE.NS`, `INFY.NS`, `TCS.NS`""")


# Weight sliders - reads from loaded portfolio
st.sidebar.subheader("🎚 Portfolio Weights (in %)")

weights = []
for i, t in enumerate(tickers_list):
    if st.session_state["loaded_weights"] is not None and i < len(st.session_state["loaded_weights"]):
        default_weight = st.session_state["loaded_weights"][i] * 100
    else:
        default_weight = round(100 / len(tickers_list), 2)
    
    w = st.sidebar.number_input(
        f"Weight for {t}",
        min_value=0.0,
        max_value=100.0,
        value=default_weight
    )
    weights.append(w)

# Clear loaded weights after use
if st.session_state["loaded_weights"] is not None:
    st.session_state["loaded_weights"] = None

weights = [w / 100 for w in weights]

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
# WEIGHT VALIDATION
# ============================================
if abs(sum(weights) - 1) > 0.001:
    st.sidebar.warning("⚠️ Weights must add up to 100%. Adjust them to continue.")
else:
    st.sidebar.caption("✅ Total: 100%")

# ============================================
# DOWNLOAD DATA
# ============================================
raw = yf.download(
    tickers_list,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

risk_free = st.sidebar.number_input(
    "Risk-free rate (%)",
    value=6.0,
    step=0.25
) / 100

# ============================================
# TABS
# ============================================
tab_overview, tab_risk, tab_benchmark, tab_monte, tab_optimizer = st.tabs(
    ["📊 Overview", "⚠️ Risk", "📈 Benchmark", "🎲 Monte Carlo" , "Optimizer"]
)

# ============================================
# OVERVIEW TAB
# ============================================
with tab_overview:
    st.subheader("📈 Price History")
    data = raw["Close"]

    if "Adj Close" in raw.columns:
        data = raw["Adj Close"]

    data = data.dropna()

    price_df = data.reset_index().melt('Date', var_name='Ticker', value_name='Price')

    chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X('Date:T', axis=alt.Axis(format='%b %Y', labelAngle=0, labelOverlap=True)),
            y=alt.Y('Price:Q'),
            color=alt.Color('Ticker:N', legend=alt.Legend(orient='bottom', title=None))
        )
        .properties(height=420, width=1000)
    )
    st.altair_chart(chart, use_container_width=False)

    st.subheader("📊 Daily Returns")
    returns = data.pct_change().dropna()
    st.dataframe(returns.tail().style.format("{:.2%}"))

if returns.empty:
    st.warning("No price data available for the selected tickers/date range.")
    st.stop()

if returns.shape[1] != len(weights):
    weights = np.repeat(1/returns.shape[1], returns.shape[1])

portfolio_return = returns.dot(weights)
cumulative = (1 + portfolio_return).cumprod()

# ============================================
# RISK TAB
# ============================================
with tab_risk:
    st.header("📉 Portfolio Risk Metrics")
    trading_days = 252

    mu_annual = portfolio_return.mean() * trading_days
    sigma_annual = portfolio_return.std() * np.sqrt(trading_days)
    sharpe = (mu_annual - risk_free) / sigma_annual
    volatility = sigma_annual * 100

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    var_95 = portfolio_return.quantile(0.05) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility (Annualized)", f"{volatility:.2f}%")
    col1.caption("Measures how **choppy** returns are. Higher = riskier.")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.caption("Return earned **per unit of risk**. >1 is generally strong.")
    col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    col3.caption("Worst peak-to-trough fall – shows **pain during crashes**.")
    col4.metric("VaR (95%)", f"{var_95:.2f}%")
    col4.caption("On a bad day, you may lose **about this much or worse** (5% chance).")

    
    # --- Correlation Matrix Heatmap ---
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

        # Melt into tidy long form (no duplicate 'Ticker' columns)
        corr_df = corr_matrix.reset_index().melt(
            id_vars="Ticker1",
            var_name="Ticker2",
            value_name="Correlation",
        )

        # Build heatmap
        heatmap = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(
                x=alt.X("Ticker1:N", sort=None, title=None),
                y=alt.Y("Ticker2:N", sort=None, title=None),
                color=alt.Color("Correlation:Q",
                                scale=alt.Scale(scheme="redgrey", domain=[-1, 1])),
                tooltip=[
                    alt.Tooltip("Ticker1:N", title="Asset A"),
                    alt.Tooltip("Ticker2:N", title="Asset B"),
                    alt.Tooltip("Correlation:Q", format=".2f"),
                ],
            )
            .properties(width=500, height=500, title="Correlation heatmap")
        )

        # Optional: overlay labels for exact values
        labels = (
            alt.Chart(corr_df)
            .mark_text(baseline="middle", align="center", fontSize=13)
            .encode(
                x="Ticker1:N",
                y="Ticker2:N",
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "datum.Correlation > 0.7 || datum.Correlation < -0.7",
                    alt.value("white"),  # white text on dark blocks
                    alt.value("black")   # black text on light blocks
                )
            )
        )


        st.altair_chart(heatmap + labels, use_container_width=True)



    st.subheader("🔄 Rolling Risk (Volatility & Sharpe)")
    st.caption("Rolling metrics show **how risk and performance change over time**, instead of one static value.")
    st.sidebar.subheader("🔍 Rolling Window Settings")

    window = st.sidebar.slider(
        "Rolling window (trading days)",
        min_value=20,
        max_value=252,
        value=60,
        step=5
    )

    rolling_std = portfolio_return.rolling(window).std() * np.sqrt(trading_days)
    rolling_vol = rolling_std * 100

    rolling_mean = portfolio_return.rolling(window).mean() * trading_days
    rolling_sharpe = (rolling_mean - risk_free) / rolling_std

    vol_df = rolling_vol.reset_index()
    vol_df.columns = ["Date", "Rolling Volatility (%)"]
    chart_vol = (
        alt.Chart(vol_df)
        .mark_line(color="#e63946")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y="Rolling Volatility (%):Q"
        )
        .properties(title="Rolling Annualized Volatility", height=320, width=950)
    )
    st.altair_chart(chart_vol, use_container_width=False)

    sharpe_df = rolling_sharpe.reset_index()
    sharpe_df.columns = ["Date", "Rolling Sharpe"]
    chart_sharpe = (
        alt.Chart(sharpe_df)
        .mark_line(color="#1d3557")
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
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
with tab_benchmark:
    st.header("📊 Portfolio vs Benchmark")
    st.sidebar.subheader("📌 Benchmark Comparison")

    benchmark_symbol = st.sidebar.selectbox(
        "Select benchmark index:",
        ("^NSEI", "^GSPC", "^NDX", "^BSESN"),
        index=0
    )

    benchmark_names = {
        "^NSEI": "Nifty 50",
        "^GSPC": "S&P 500",
        "^NDX": "Nasdaq 100",
        "^BSESN": "Sensex"
    }
    benchmark_label = benchmark_names.get(benchmark_symbol, "Benchmark")

    # Use the user-selected start_date for benchmark
    benchmark = yf.download(
        benchmark_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    # Benchmark cumulative for charting
    # 1. Calculate Daily Returns
    benchmark_returns = benchmark.pct_change().dropna()

    # 2. Align Daily Returns FIRST (This is the fix: Resets both to start line)
    # portfolio_return comes from the Overview tab
    aligned_df = pd.concat([portfolio_return, benchmark_returns], axis=1).dropna()
    aligned_df.columns = ["Portfolio", benchmark_label]

    # 3. Calculate Cumulative Growth on aligned data (Now both start at 1.0)
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
        "Number of days to simulate (trading days)",
        min_value=60, max_value=756, value=252, step=10
    )
    num_sims = st.slider(
        "Number of simulations",
        min_value=100, max_value=3000, value=200, step=100
    )

    @st.cache_data(show_spinner=False)
    def run_simulation(mu, sigma, num_days, num_sims):
        start_value = 1
        sim_results = np.zeros((num_days, num_sims))
        for s in range(num_sims):
            daily_random_returns = np.random.normal(mu, sigma, num_days)
            sim_path = start_value * np.cumprod(1 + daily_random_returns)
            sim_results[:, s] = sim_path
        return sim_results

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

    lower_line = alt.Chart(percentile_df).mark_line(color="#d61f1f").encode(
        x="Day:Q", y=alt.Y("P5:Q", axis=alt.Axis(title="Portfolio Value (x)"))
    )
    upper_line = alt.Chart(percentile_df).mark_line(color="#4ade80").encode(
        x="Day:Q", y="P95:Q"
    )
    median_line = alt.Chart(percentile_df).mark_line(color="orange", size=2).encode(
        x="Day:Q", y="P50:Q"
    )
    chart_mc = alt.Chart(sim_melted).mark_line(opacity=0.1).encode(
        x="Day:Q", y="Value:Q"
    ).properties(height=420, width=900)

    final_chart = lower_line + upper_line + chart_mc + median_line
    st.altair_chart(final_chart, use_container_width=False)

    st.subheader("Monte Carlo Summary (End of Period)")
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
    over the next {num_days} days.
    """)
    st.caption("""
    These estimates come from Monte Carlo simulations based on historical volatility 
    and returns. They do **not** guarantee outcomes, but help visualize risk and uncertainty.
    """)

# ============================================
# OPTIMIZER TAB
# ============================================

st.sidebar.subheader("Optimizer Settings")
lower_limit = st.sidebar.number_input(
            "Min weight per asset (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step = 1.0,
            )

with tab_optimizer:
    from scipy.optimize import minimize

    st.header("🔧 Portfolio Optimizer – Max Sharpe")

    def portfolio_return_opt(weights, mean_returns):
        return np.dot(weights, mean_returns)

    def portfolio_volatility_opt(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free):
        ret = portfolio_return_opt(weights, mean_returns)
        vol = portfolio_volatility_opt(weights, cov_matrix)
        return -((ret - risk_free) / vol)


    def max_sharpe_portfolio(mean_returns, cov_matrix, risk_free):
        num_assets = len(mean_returns)
        init_weights = np.array(num_assets * [1.0 / num_assets])
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
        bounds = tuple((lower_limit / 100, 1) for _ in range(num_assets))

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
        sharpe = (ret - risk_free) / vol
        return opt_weights, ret, vol, sharpe

    mean_returns = returns.mean() * 252
    cov_matrix   = returns.cov() * 252


    # Volatility Minimizer
    def min_volatility_portfolio(mean_returns, cov_matrix):
        num_assets = len(mean_returns)
        init_weights = np.array(num_assets * [1.0 / num_assets])
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
        bounds = tuple((lower_limit / 100, 1) for _ in range(num_assets))

        # Note: We directly minimize 'portfolio_volatility_opt'
        result = minimize(
            portfolio_volatility_opt,
            init_weights,
            args=(cov_matrix,),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result

    # 🟢 NEW: Wrapper to get stats
    def get_min_volatility_portfolio(mean_returns, cov_matrix, risk_free):
        result = min_volatility_portfolio(mean_returns, cov_matrix)
        opt_weights = result.x
        
        # Calculate stats for the new weights
        ret = portfolio_return_opt(opt_weights, mean_returns)
        vol = portfolio_volatility_opt(opt_weights, cov_matrix)
        sharpe = (ret - risk_free) / vol if vol != 0 else 0
        
        return opt_weights, ret, vol, sharpe

# --- 3. Strategy Selection Buttons (Side-by-Side) ---
    # 1. Select Strategy
    goal = st.radio(
        "Optimization Goal:",
        ["🚀 Max Sharpe (Growth)", "🛡️ Min Volatility (Safety)"],
        horizontal=True
    )

    # 2. Single Run Button
    if st.button("Run Optimization", type="primary"):
        if "Max Sharpe" in goal:
            # Run Max Sharpe
            opt_weights_calc, ret, vol, sharpe = get_max_sharpe_portfolio(
                mean_returns, cov_matrix, risk_free
            )
        else:
            # Run Min Volatility (Make sure this function is in optimizer.py)
            opt_weights_calc, ret, vol, sharpe = get_min_volatility_portfolio(
                mean_returns, cov_matrix, risk_free
            )
        
        # Save results to session state
        st.session_state["opt_weights"] = opt_weights_calc
        st.session_state["opt_stats"] = (ret, vol, sharpe)

        if "opt_weights" in st.session_state and "opt_stats" in st.session_state:
            # Convert to NumPy array so .shape works
            opt_weights_display = np.array(st.session_state["opt_weights"])
            ret, vol, sharpe = st.session_state["opt_stats"]

        if opt_weights_display.shape[0] != mean_returns.shape[0]:
            st.warning("Optimizer weights were from an older ticker list – Optimize Again!")
            opt_weights_display = np.repeat(1/mean_returns.shape[0], mean_returns.shape[0])
            st.session_state["opt_weights"] = opt_weights_display



        def portfolio_stats(weights, mean_returns, cov_matrix, risk_free):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (port_return - risk_free) / port_vol if port_vol != 0 else 0
            return port_return, port_vol, sharpe

        manual_weights_for_compare = []
        for t in tickers_list:
            w = st.session_state.get(f"number_input_{t}", round(100 / len(tickers_list), 2))
            manual_weights_for_compare.append(w / 100)

        if opt_weights_display.shape[0] != mean_returns.shape[0]:
            st.warning("Optimizer weights were from an older ticker list – Optimize Again!")
            opt_weights_display = np.repeat(1/mean_returns.shape[0], mean_returns.shape[0])
            st.session_state["opt_weights"] = opt_weights_display

        opt_ret, opt_vol, opt_sharpe = portfolio_stats(
            opt_weights_display, mean_returns, cov_matrix, risk_free
        )
        
        manual_ret, manual_vol, manual_sharpe = portfolio_stats(
            np.array(manual_weights_for_compare), mean_returns, cov_matrix, risk_free
        )

        st.subheader("📊 Manual vs Optimized Portfolio (Annualized)")
        compare_df = pd.DataFrame({
            "Metric": ["Expected Return", "Volatility (Risk)", "Sharpe Ratio"],
            "Manual": [f"{manual_ret*100:.2f}%", f"{manual_vol*100:.2f}%", f"{manual_sharpe:.2f}"],
            "Optimized": [f"{opt_ret*100:.2f}%", f"{opt_vol*100:.2f}%", f"{opt_sharpe:.2f}"]
        }, index=[1, 2, 3])
        st.table(compare_df)

        st.subheader("Optimized Portfolio Weights")
        opt_df = pd.DataFrame({
            "S.No": np.arange(1, len(mean_returns) + 1),
            "Asset": mean_returns.index,
            "Weight %": (opt_weights_display * 100).round(2)
        })
        total_row = pd.DataFrame([["TOTAL", "—", opt_df["Weight %"].sum().round(2)]],
                                 columns=["S.No", "Asset", "Weight %"])
        opt_df = pd.concat([opt_df, total_row], ignore_index=True)
        st.dataframe(opt_df, hide_index=True)

        if st.button("✅ Apply These Weights", type="primary"):
            st.session_state["use_optimized"] = True
            st.success("✅ Optimized weights applied!")
            st.rerun()

# ============================================
# SIDEBAR: SAVE/LOAD PORTFOLIOS
# ============================================
st.sidebar.markdown("---")

# Load Portfolio Section
try:
    all_portfolios = portfolios_ws.get_all_records()
    
    # Filter user portfolios safely
    user_portfolios = [
        p for p in all_portfolios 
        if st.session_state.get("user_email") and 
           p.get("email", "").lower() == st.session_state["user_email"].lower()
    ]

    if user_portfolios:
        st.sidebar.subheader("📂 Saved Portfolios")
        
        portfolio_options = [f"{p['portfolio_name']}" for p in user_portfolios]
        
        selected_display = st.sidebar.selectbox(
            "Choose a portfolio:",
            portfolio_options
        )
        
        if st.sidebar.button("🚀 Load Portfolio"):
            selected_index = portfolio_options.index(selected_display)
            chosen = user_portfolios[selected_index]
            
            loaded_tickers = chosen["tickers"].split(",")
            loaded_weights = [float(w) for w in chosen["weight"].split(",")]
            
            st.session_state["loaded_tickers"] = loaded_tickers
            st.session_state["loaded_weights"] = loaded_weights
            
            if chosen.get("optimized_weights"):
                opt_weights_str = chosen["optimized_weights"]
                if opt_weights_str:
                    st.session_state["opt_weights"] = [float(w) for w in opt_weights_str.split(",")]
                    st.session_state["use_optimized"] = True
            
            st.sidebar.success(f"✅ Loading {chosen['portfolio_name']}...")
            st.rerun()

except Exception as e:
    st.sidebar.error(f"Error loading portfolios: {e}")

# Save Portfolio
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
            st.session_state.get("opt_weights") if st.session_state["use_optimized"] else None
        )
        st.sidebar.success("✅ Saved!")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")



# ============================================
# FOOTER & DISCLAIMER
# ============================================

# 1. The "Harbhajan The Great" Footer (Fixed Bottom Right)
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 70px;
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
        pointer-events: none; /* Let clicks pass through if needed */
    }
    </style>
    <div class="footer">Harbhajan The Great</div>
    """,
    unsafe_allow_html=True
)

# 2. The Disclaimer Button (Fixed Below Footer)
# We use HTML <details> so it toggles instantly without re-running Python
st.markdown(
    """
    <style>
    /* Container aligns the button with the footer */
    .disclaimer-container {
        position: fixed;
        bottom: 25px;
        right: 15px;
        z-index: 1001;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
    }
    
    /* The Button styling */
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
        list-style: none; /* Hides default arrow */
        transition: background 0.2s;
    }
    .disclaimer-btn:hover {
        background-color: #d9002a;
    }
    /* Hide default marker in Chrome/Safari */
    .disclaimer-btn::-webkit-details-marker {
        display: none;
    }

    /* The Popup Box styling */
    .disclaimer-content {
        position: absolute;
        bottom: 35px; /* Pops up above the button */
        right: 0;
        width: 280px;
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
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>

    <div class="disclaimer-container">
        <details>
            <summary class="disclaimer-btn">⚠️ Disclaimer</summary>
            <div class="disclaimer-content">
                <strong>Legal Notice:</strong><br>
                This dashboard is for educational purposes only. 
                It does not constitute investment advice. 
                Market data may be delayed or inaccurate. 
                Users are solely responsible for financial decisions made using this tool.
            </div>
        </details>
    </div>
    """,
    unsafe_allow_html=True
)