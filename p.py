import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(layout="wide")
st.title("📊 Portfolio Performance & Risk Dashboard")
st.write("Analyze returns, risk, and benchmark performance in one place.")

# =========================
# SIDEBAR — INPUTS
# =========================
st.sidebar.header("Portfolio Inputs")

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    value="AAPL,MSFT,AMZN,META,GOOG"
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date   = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

tickers_list = [t.strip() for t in tickers.split(",")]

st.sidebar.subheader("🎚 Portfolio Weights (in %)")

weights = []
for i, t in enumerate(tickers_list):
    key = f"weight_{t}"
    default_val = round(100 / len(tickers_list), 2)

    if "active_weights" in st.session_state:
        default_val = round(st.session_state["active_weights"][i] * 100, 2)

    w = st.sidebar.number_input(
        f"Weight for {t}",
        min_value=0.0,
        max_value=100.0,
        value=default_val,
        key=key
    )
    weights.append(w)

weights = [w / 100 for w in weights]

if abs(sum(weights) - 1) > 0.001:
    st.warning("⚠️ Weights must add up to 100%. Adjust them to continue.")

# Risk-free — global
st.sidebar.subheader("💰 Risk-Free Rate (annual)")
risk_free = st.sidebar.number_input("Risk-free rate (%)", value=6.0, step=0.25) / 100

# =========================
# DATA DOWNLOAD
# =========================
raw = yf.download(
    tickers_list,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

tab_overview, tab_risk, tab_benchmark, tab_monte, tab_optimizer = st.tabs(
    ["📊 Overview", "⚠️ Risk", "📈 Benchmark", "🎲 Monte Carlo", "Optimizer"]
)

# =========================
# PRICE / RETURNS
# =========================
data = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
data = data.dropna()
returns = data.pct_change().dropna()

if returns.empty:
    st.warning("No price data available for the selected tickers/date range.")
    st.stop()

# equal-weight fallback ONLY (no metrics here)
if returns.shape[1] != len(weights):
    weights = np.repeat(1 / returns.shape[1], returns.shape[1])

# =========================
# ACTIVE WEIGHTS LOGIC
# =========================
active_weights = st.session_state.get("active_weights", weights)
active_weights = np.array(active_weights)

if active_weights.shape[0] != returns.shape[1]:
    active_weights = np.repeat(1/returns.shape[1], returns.shape[1])
    st.info("Ticker list changed — weights reset to equal.")

portfolio_return = returns.dot(active_weights)
cumulative = (1 + portfolio_return).cumprod()

# =========================
# OVERVIEW TAB
# =========================
with tab_overview:
    st.subheader("📈 Price History")

    price_df = data.reset_index().melt("Date", var_name="Ticker", value_name="Price")

    chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y="Price:Q",
            color=alt.Color("Ticker:N", legend=alt.Legend(orient="bottom", title=None))
        )
        .properties(height=420, width=1000)
    )
    st.altair_chart(chart, use_container_width=False)

    st.subheader("📊 Daily Returns")
    st.dataframe(returns.tail().style.format("{:.2%}"))

# =========================
# RISK TAB
# =========================
with tab_risk:
    st.header("📉 Portfolio Risk Metrics")

    trading_days = 252
    annual_return = portfolio_return.mean() * trading_days
    annual_vol    = portfolio_return.std() * np.sqrt(trading_days)
    volatility    = annual_vol * 100
    sharpe        = (annual_return - risk_free) / annual_vol

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    var_95 = portfolio_return.quantile(0.05) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Volatility (Annualized)", f"{volatility:.2f}%")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    col4.metric("VaR (95%)", f"{var_95:.2f}%")

    st.subheader("🔄 Rolling Risk (Volatility & Sharpe)")
    window = st.sidebar.slider("Rolling window (trading days)", 20, 252, 60, step=5)

    rolling_vol = (portfolio_return.rolling(window).std() * np.sqrt(252)) * 100
    rolling_sharpe = (
        (portfolio_return.rolling(window).mean() * 252 - risk_free) /
        (portfolio_return.rolling(window).std() * np.sqrt(252))
    )

    vol_df = rolling_vol.reset_index().rename(columns={0: "Rolling Volatility (%)"})
    sharpe_df = rolling_sharpe.reset_index().rename(columns={0: "Rolling Sharpe"})

    st.altair_chart(
        alt.Chart(vol_df).mark_line(color="#e63946").encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y="Rolling Volatility (%)"
        ).properties(height=320, width=950),
        use_container_width=False
    )

    st.altair_chart(
        alt.Chart(sharpe_df).mark_line(color="#1d3557").encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y="Rolling Sharpe"
        ).properties(height=320, width=950),
        use_container_width=False
    )

# =========================
# BENCHMARK TAB
# =========================
with tab_benchmark:
    st.header("📊 Portfolio vs Benchmark")
    st.sidebar.subheader("📌 Benchmark Comparison")

    benchmark_symbol = st.sidebar.selectbox(
        "Select benchmark index:",
        ("^NSEI", "^GSPC", "^NDX", "^BSESN"),
        index=0
    )

    names = {"^NSEI": "Nifty 50", "^GSPC": "S&P 500", "^NDX": "Nasdaq 100", "^BSESN": "Sensex"}
    label = names.get(benchmark_symbol, "Benchmark")

    bench = yf.download(
        benchmark_symbol, start=start_date, end=end_date, auto_adjust=True, progress=False
    )["Close"]

    bench_cum = (1 + bench.pct_change().dropna()).cumprod()

    portfolio_df = cumulative.to_frame(name="Portfolio")
    benchmark_df = bench_cum.to_frame(name=label)

    compare_df = (
        pd.concat([portfolio_df, benchmark_df], axis=1)
        .dropna()
        .reset_index()
    )

    chart_compare = (
        alt.Chart(compare_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", axis=alt.Axis(format="%b %Y", labelAngle=0)),
            y=alt.Y("value:Q", title="Growth Index"),
            color=alt.Color("variable:N", legend=alt.Legend(title=None))
        )
        .transform_fold(["Portfolio", label], as_=["variable", "value"])
        .properties(height=420, width=1000)
    )

    st.altair_chart(chart_compare, use_container_width=False)

    alpha = (cumulative.iloc[-1] - bench_cum.iloc[-1]) * 100
    color = "green" if alpha >= 0 else "red"

    st.markdown(
        f"<h4>📈 Alpha vs {label}: <span style='color:{color}'>{float(alpha):.2f}%</span></h4>",
        unsafe_allow_html=True
    )

# =========================
# MONTE CARLO TAB
# =========================
with tab_monte:
    st.header("🎲 Monte Carlo Simulation")
    daily_returns = portfolio_return
    mu, sigma = daily_returns.mean(), daily_returns.std()

    num_days = st.slider("Number of days", 60, 756, 252, step=10)
    num_sims = st.slider("Simulations", 100, 3000, 200, step=100)

    @st.cache_data
    def run_sim(mu, sigma, days, sims):
        sim = np.zeros((days, sims))
        for s in range(sims):
            dr = np.random.normal(mu, sigma, days)
            sim[:, s] = np.cumprod(1 + dr)
        return sim

    sim_results = run_sim(mu, sigma, num_days, num_sims)

    st.subheader("Simulated Future Portfolio Paths")
    sample = min(60, num_sims)

    sim_df = pd.DataFrame(sim_results[:, :sample])
    sim_df["Day"] = np.arange(1, num_days + 1)
    sim_melt = sim_df.melt(id_vars="Day", value_name="Value")

    p5  = np.percentile(sim_results, 5, axis=_
