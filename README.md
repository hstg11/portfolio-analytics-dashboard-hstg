# 📊 Portfolio Analytics Dashboard — Risk, Monte Carlo, Optimization, Efficient Frontier & AI Insights (Streamlit)

A production-style **Portfolio Analytics Dashboard** built with **Python + Streamlit**, aimed at **quant-enabled portfolio analysis** and **equity research workflows**.

Users can input a portfolio, analyze performance + risk, compare with benchmarks, run Monte Carlo simulations, optimize allocations using **Markowitz Mean–Variance (MPT)**, visualize the **Efficient Frontier**, and generate **AI-driven portfolio insights** (educational only).

---

## 🚀 Live App
🔗 Streamlit: https://portfolio-analytics-dashboard-hstg.streamlit.app/
🔗 GitHub: https://github.com/hstg11/portfolio-analytics-dashboard-hstg


## ✅ Features (V1)

### 🔐 1) Signup + Persistent Login
- Mandatory signup gate: **Name, Email, Mobile, City**
- Persistent login via **cookies**
- User records stored in **Google Sheets** (lightweight DB)

---

### 📈 2) Portfolio Overview
- Multi-asset price history chart
- Daily returns table
- Clean UI designed for quick analysis

---

### ⚠️ 3) Risk & Performance Analytics
Includes:
- **Annualized volatility**
- **Sharpe ratio**
- **Sortino ratio** (downside risk-adjusted return)
- **Max drawdown**
- **VaR (95%)**
- **CVaR / TVaR** (tail risk beyond VaR)
- Rolling volatility + rolling Sharpe
- Correlation heatmap (diversification insight)

---

### 📊 4) Benchmark Comparison + Beta
- Benchmark comparison vs:
  - **Nifty 50**, **Sensex**
  - **S&P 500**
  - **Nasdaq 100**
- Alpha tracking
- **Portfolio beta vs benchmark** (market sensitivity)

---

### 🎲 5) Monte Carlo Simulation
- Simulates 100s–3000s of future portfolio paths for 60 to 756 days using historical μ and σ
- Shows percentile bands **P5 / P50 / P95**
- Helps visualize outcome uncertainty (not prediction)

---

### 🧠 6) Portfolio Optimizer (Markowitz MPT)
Supports:
- 🚀 **Max Sharpe Portfolio**
- 🛡️ **Min Volatility Portfolio**
- Min weight constraint per asset
- Manual vs optimized comparison
- Apply optimized weights button directly to the dashboard

---

### 🏔️ 7) Efficient Frontier (Vision Frontier)
- Generates **500-5000 random portfolios**
- Plots risk-return scatter based on inputed risk appetite
- Highlights:
  - Current portfolio
  - Max Sharpe portfolio
  - Efficient Frontier curve (best achievable set)

---

### 🤖 8) AI Insight Layer (Gemini)
- Short **AI Quick Summary** in sidebar
- Long institutional-style analysis of 1000 words (multi-section report)
- API key rotation support to handle free quotas(Gemini)
- Text to Speech
- Outputs are **educational only**, not advisory

---

## 🧠 Why This Project is Different
Most student finance projects stop at basic return charts.

This app includes institutional-grade additions:
- Tail risk (CVaR), downside risk (Sortino), drawdowns
- Efficient frontier visualization
- Optimization with apply bridge into UI
- Persistent users + database logging
- AI portfolio commentary layer

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Market Data:** yFinance
- **Core Analytics:** Pandas, NumPy
- **Charts:** Altair
- **Optimization:** SciPy (SLSQP)
- **Persistence / DB:** Google Sheets (gspread)
- **Auth:** session_state + cookies
- **AI:** Gemini 2.5 Flash API (key rotation)

---

## ⚙️ Run Locally

### 1) Clone repo
```bash
git clone https://github.com/hstg11/portfolio-analytics-dashboard-hstg.git
cd portfolio-analytics-dashboard-hstg
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run Streamlit

```bash
streamlit run app.py
```

---

## 🔐 Secrets / Credentials Setup (Important)

This app uses:

* **Google Service Account** credentials
* Streamlit secrets via `.streamlit/secrets.toml` OR Streamlit Cloud secrets

Example secret structure:

```toml
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
token_uri = "https://oauth2.googleapis.com/token"
```

✅ DO NOT commit:

* `.env`
* `.streamlit/secrets.toml`
* API keys

---

## 📌 NSE Stock Input Note

For Indian stocks:

* Use `.NS` for NSE tickers or `.BO` for BSE tickers 
  Example: `500325.BO`(reliance), `INFY.NS`, `TCS.NS`

---

## ⚠️ Disclaimer

This dashboard is for **educational and research purposes only**.
It does **not** provide investment advice or recommendations.

Markets carry risk. Users are solely responsible for decisions made using this tool.

---

## 🧩 Roadmap (V2 Ideas)

Potential upgrades:

* Fama-French factor exposure
* Risk attribution / MCTR
* Return attribution
* Event stress testing (COVID / 2022 tightening)
* PDF export (memo-style report)

---

## 👤 Author

**Harbhajan Singh Tuteja**
Portfolio analytics + quant-enabled finance projects
LinkedIn: https://www.linkedin.com/in/harbhajan-singh-tuteja/


---

## ⭐ Credits

* Streamlit (UI)
* yFinance (market data)
* SciPy (optimization)
* Google Sheets API (data persistence)
* Gemini API (AI insights)
