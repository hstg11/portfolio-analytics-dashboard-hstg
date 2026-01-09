#app.py
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

# Persistent portfolio state
if "current_tickers" not in st.session_state:
    st.session_state["current_tickers"] = "AAPL,MSFT,AMZN,META,GOOG"

if "current_weights_pct" not in st.session_state:
    st.session_state["current_weights_pct"] = None

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

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

tickers_list = [t.strip() for t in tickers.split(",")]

tickers_list = [t.strip() for t in tickers.split(",")]

# ✅ Detect if ticker count changed - reset to equal weights
if "last_ticker_count" not in st.session_state:
    st.session_state["last_ticker_count"] = len(tickers_list)

if len(tickers_list) != st.session_state["last_ticker_count"]:
    # Ticker count changed! Reset weights
    st.session_state["last_ticker_count"] = len(tickers_list)
    st.session_state["current_weights_pct"] = None  # Force equal weights

st.sidebar.info("""Input Guide for Indian Stocks - NSE stocks → (Ticker + `.NS`) , e.g. `RELIANCE.NS`, `INFY.NS`, `TCS.NS`""")


# Weight sliders - reads from loaded portfolio
st.sidebar.subheader("🎚 Portfolio Weights (in %)")

weights = []
for i, t in enumerate(tickers_list):
    if st.session_state["loaded_weights"] is not None and i < len(st.session_state["loaded_weights"]):
        default_weight = st.session_state["loaded_weights"][i] * 100
    elif st.session_state["current_weights_pct"] and i < len(st.session_state.get("current_weights_pct", [])):
        # ✅ Only use stored weights if count matches
        if len(st.session_state["current_weights_pct"]) == len(tickers_list):
            default_weight = st.session_state["current_weights_pct"][i]
        else:
            default_weight = round(100 / len(tickers_list), 2)
    else:
        default_weight = round(100 / len(tickers_list), 2)
    
    w = st.sidebar.number_input(
        f"Weight for {t}",
        min_value=0.0,
        max_value=100.0,
        value=default_weight
    )
    weights.append(w)

st.session_state["current_weights_pct"] = weights.copy()

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

    # Use helper from portfolio.py
    metrics = risk_metrics(portfolio_return, cumulative, risk_free)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility (Annualized)", f"{metrics['volatility_pct']:.2f}%")
    col1.caption("Measures how **choppy** returns are. Higher = riskier.")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    col2.caption("Return earned **per unit of risk**. >1 is generally strong.")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    col3.caption("Worst peak-to-trough fall – shows **pain during crashes**.")
    col4.metric("VaR (95%)", f"{metrics['var_95_pct']:.2f}%")
    col4.caption("On a bad day, you may lose **about this much or worse** (5% chance).")

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

        # Build heatmap
        heatmap = (
            alt.Chart(corr_df)
            .mark_rect()
            .encode(
                x=alt.X("Ticker1:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
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

        # Overlay labels
        labels = (
            alt.Chart(corr_df)
            .mark_text(baseline="middle", align="center", fontSize=13)
            .encode(
                x="Ticker1:N",
                y="Ticker2:N",
                text=alt.Text("Correlation:Q", format=".2f"),
                color=alt.condition(
                    "datum.Correlation > 0.7 || datum.Correlation < -0.7",
                    alt.value("white"),
                    alt.value("black")
                )
            )
        )

        st.altair_chart(heatmap + labels, use_container_width=True)

    
    st.subheader("🔄 Rolling Risk (Volatility & Sharpe)")
    st.caption("Rolling metrics show **how risk and performance change over time**.")

    st.sidebar.subheader("🔍 Rolling Window Settings")
    window = st.sidebar.slider(
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
    # Use helper from portfolio.py instead of inline download
    benchmark_returns, bench_cum = benchmark_series(benchmark_symbol, start_date, end_date)

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
st.sidebar.subheader("Optimizer Settings")
lower_limit = st.sidebar.number_input(
    "Min weight per asset (%)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
)

with tab_optimizer:
    st.header("🔧 Portfolio Optimizer")

    # --- 1. Calculations ---
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # --- 2. Strategy Selection ---
    goal = st.radio(
        "Optimization Goal:",
        ["🚀 Max Sharpe (Growth)", "🛡️ Min Volatility (Safety)"],
        horizontal=True
    )

    if st.button("Run Optimization", type="primary"):
        if "Max Sharpe" in goal:
            opt_weights_calc, ret, vol, sharpe = get_max_sharpe_portfolio(
                mean_returns, cov_matrix, risk_free, lower_limit
            )
        else:
            opt_weights_calc, ret, vol, sharpe = get_min_volatility_portfolio(
                mean_returns, cov_matrix, risk_free, lower_limit
            )
        
        # Save results
        st.session_state["opt_weights"] = opt_weights_calc
        st.session_state["opt_stats"] = (ret, vol, sharpe)
        st.rerun()

    # --- 3. Display Results ---
    if "opt_weights" in st.session_state and "opt_stats" in st.session_state:
        opt_weights_display = np.array(st.session_state["opt_weights"])
        ret, vol, sharpe = st.session_state["opt_stats"]

        # Safety Check: Ticker mismatch
        if opt_weights_display.shape[0] != mean_returns.shape[0]:
            st.warning("⚠️ Ticker list changed. Please click 'Run Optimization' again.")
            opt_weights_display = np.repeat(1/mean_returns.shape[0], mean_returns.shape[0])
            st.session_state["opt_weights"] = opt_weights_display
        else:
            # Manual weights for comparison
            manual_weights_array = np.array(weights)

            # Calculate stats using optimizer.py helper
            opt_ret, opt_vol, opt_sharpe = portfolio_stats(
                opt_weights_display, mean_returns, cov_matrix, risk_free
            )
            manual_ret, manual_vol, manual_sharpe = portfolio_stats(
                manual_weights_array, mean_returns, cov_matrix, risk_free
            )

            # --- Comparison Table ---
            st.subheader("📊 Manual vs Optimized Portfolio (Annualized)")
            compare_df = pd.DataFrame({
                "Metric": ["Expected Return", "Volatility (Risk)", "Sharpe Ratio"],
                "Manual": [f"{manual_ret*100:.2f}%", f"{manual_vol*100:.2f}%", f"{manual_sharpe:.2f}"],
                "Optimized": [f"{opt_ret*100:.2f}%", f"{opt_vol*100:.2f}%", f"{opt_sharpe:.2f}"]
            }, index=[1, 2, 3])
            st.table(compare_df)

            # --- Charts (unchanged) ---
                        # cumulative returns chart
            manual_daily_ret = returns.dot(manual_weights_array)
            opt_daily_ret = returns.dot(opt_weights_display)

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
                x=alt.X("Date:T", axis=alt.Axis(format='%b %Y', title=None)),
                y=alt.Y("Growth Index:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Portfolio:N", scale=alt.Scale(
                    domain=['Manual Portfolio', 'Optimized Portfolio'], 
                    range=['#cccccc', '#4ade80']
                )),
                tooltip=["Date:T", "Portfolio", alt.Tooltip("Growth Index", format=".2f")]
            ).properties(height=350, width=800)
            st.altair_chart(perf_chart, use_container_width=True)
            st.caption("This chart shows **historical backtested performance** based on your chosen period. It is not a forecast.")

            # weight allocation chart
            alloc_df = pd.DataFrame({
                "Ticker": tickers_list,
                "Manual": manual_weights_array * 100,
                "Optimized": opt_weights_display * 100
            }).melt("Ticker", var_name="Type", value_name="Weight (%)")

            alloc_chart = alt.Chart(alloc_df).mark_bar().encode(
                x=alt.X("Ticker:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Weight (%):Q"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=['Manual', 'Optimized'], 
                    range=['#cccccc', '#4ade80']
                )),
                xOffset="Type:N",
                tooltip=["Ticker", "Type", alt.Tooltip("Weight (%)", format=".1f")]
            ).properties(height=350)
            
            st.altair_chart(alloc_chart, use_container_width=True)

            # weights table
            st.subheader("Optimized Portfolio Weights")
            opt_df = pd.DataFrame({
                "Asset": mean_returns.index,
                "Weight %": (opt_weights_display * 100).round(2)
            })
            st.dataframe(opt_df, hide_index=True)

            # apply button
            if st.button("✅ Apply These Weights", type="primary"):
                st.session_state["use_optimized"] = True
                st.session_state["opt_weights"] = opt_weights_display.tolist() 
                st.success("✅ Optimized weights applied!")
                st.toast("✅ Optimized weights applied!", icon="✅")
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
    <div class="footer">Harbhajan The Great</div>
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
    All charts and metrics are for illustrative purposes only. They should not be interpreted as financial advice or guarantees of future outcomes.
    </div>
    </details>
    </div>
    """,
    unsafe_allow_html=True
)