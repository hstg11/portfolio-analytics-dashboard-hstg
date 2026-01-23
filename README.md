# 📊 Portfolio Analytics Dashboard — Risk, Monte Carlo, Optimization, Efficient Frontier & AI Insights

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://portfolio-analytics-dashboard-hstg.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade **Portfolio Analytics Dashboard** built with **Python + Streamlit**, designed for **quantitative portfolio analysis** and **equity research workflows**.

[Live Demo](https://portfolio-analytics-dashboard-hstg.streamlit.app/) • [Report Bug](https://github.com/hstg11/portfolio-analytics-dashboard-hstg/issues) • [Request Feature](https://github.com/hstg11/portfolio-analytics-dashboard-hstg/issues)

</div>

---

## 🎯 Overview

This dashboard enables users to:
- ✅ Input custom portfolios and analyze performance + risk metrics
- ✅ Compare against major benchmarks (Nifty 50, S&P 500, Nasdaq 100, Sensex)
- ✅ Run **Monte Carlo simulations** for future scenario analysis
- ✅ Optimize allocations using **Markowitz Mean-Variance Theory (MPT)**
- ✅ Visualize the **Efficient Frontier** with interactive risk-return mapping
- ✅ Generate **AI-driven portfolio insights** powered by Gemini (educational only)

**Built for:** Investors, researchers, finance students, and portfolio managers seeking institutional-grade analytics in an accessible format.

---

## 📸 Screenshots

### Dashboard Overview
![Portfolio Overview](screenshots/overview.png)
*Multi-asset price history with clean, professional UI*

### Risk Analytics
![Risk Metrics](screenshots/risk.png)
*Comprehensive risk metrics including VaR, CVaR, Sharpe, and Sortino ratios*

### Efficient Frontier
![Efficient Frontier](screenshots/frontier.png)
*Interactive visualization of optimal risk-return portfolios*

### AI Analysis
![AI Insights](screenshots/ai.png)
*AI-generated portfolio analysis with institutional-style reporting*

---

## ✨ Key Features

### 🔐 1. User Authentication & Persistence
- **Mandatory signup** with email, name, mobile, and city
- **Cookie-based persistent login** across sessions
- **Google Sheets backend** for lightweight user database
- Secure credential management via service accounts

### 📈 2. Portfolio Overview
- **Multi-asset price charts** with interactive Altair visualizations
- **Daily returns table** showing last 5 trading days
- **Company name resolution** (auto-fetches names from tickers)
- Support for **global markets** (NSE, BSE, NYSE, NASDAQ)

### ⚠️ 3. Advanced Risk & Performance Analytics
Comprehensive risk metrics including:
- **Annualized Volatility** — overall portfolio choppiness
- **Sharpe Ratio** — risk-adjusted return measure
- **Sortino Ratio** — downside risk-adjusted return (preferred over Sharpe)
- **Maximum Drawdown** — worst peak-to-trough decline
- **Value at Risk (VaR)** — 95% & 99% confidence levels
- **Conditional VaR (CVaR/TVaR)** — tail risk beyond VaR threshold
- **Rolling Volatility & Sharpe** — time-varying risk visualization
- **Correlation Heatmap** — diversification insights

### 📊 4. Benchmark Comparison + Beta Analysis
- Compare portfolio performance vs:
  - **Nifty 50** (^NSEI)
  - **Sensex** (^BSESN)
  - **S&P 500** (^GSPC)
  - **Nasdaq 100** (^NDX)
- **Alpha tracking** — outperformance vs benchmark
- **Portfolio Beta** — market sensitivity measure

### 🎲 5. Monte Carlo Simulation
- Simulate **100-3000+ future portfolio paths**
- Forecast horizon: **60-756 trading days**
- Uses historical mean (μ) and volatility (σ)
- Displays **percentile bands** (P5, P50, P95)
- Helps visualize **outcome uncertainty** (not predictions)

### 🧠 6. Portfolio Optimizer (Markowitz MPT)
Built-in optimization engine supporting:
- 🚀 **Max Sharpe Portfolio** — maximize risk-adjusted returns
- 🛡️ **Min Volatility Portfolio** — minimize portfolio risk
- **Minimum weight constraints** per asset
- **Side-by-side comparison** of manual vs optimized allocations
- **One-click apply** optimized weights to dashboard

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

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (Python 3.13 has compatibility issues with some dependencies)
- **Google Cloud Service Account** (for Google Sheets integration)
- **Gemini API Keys** (for AI insights - free tier available)

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/hstg11/portfolio-analytics-dashboard-hstg.git
cd portfolio-analytics-dashboard-hstg
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Configure Secrets

Create `.streamlit/secrets.toml` in your project root:

```toml
# Google Sheets API Configuration
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour-Private-Key-Here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"

# Gemini API Keys (for AI insights)
GEMINI_API_KEY_1 = "your-gemini-key-1"
GEMINI_API_KEY_2 = "your-gemini-key-2"
GEMINI_API_KEY_3 = "your-gemini-key-3"
```

**Important:** Add `.streamlit/secrets.toml` to `.gitignore` to prevent credential leaks!

#### 4. Setup Google Sheets Backend

1. Create a new Google Sheet
2. Create two worksheets:
   - `portfolio_users` — for user authentication data
   - `portfolios` — for saved portfolio snapshots
3. Share the sheet with your service account email (found in `client_email`)
4. Update sheet configuration in `config.py`:
   ```python
   SPREADSHEET_NAME = "your-sheet-name"
   USERS_WORKSHEET = "portfolio_users"
   PORTFOLIOS_WORKSHEET = "portfolios"
   ```

#### 5. Get Gemini API Keys

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create API keys (free tier: 15 requests/minute)
3. Add keys to `.streamlit/secrets.toml`

#### 6. Run Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📝 Usage Guide

### Portfolio Input

**For Indian Stocks:**
- NSE stocks: `TICKER.NS` (e.g., `INFY.NS`, `TCS.NS`)
- BSE stocks: `TICKER.BO` (e.g., `500325.BO` for Reliance)

**For US Stocks:**
- Simply use ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`)

**Example Portfolio:**
```
Tickers: AAPL, MSFT, GOOGL, INFY.NS, TCS.NS
Weights: 20%, 20%, 20%, 20%, 20%
```

### Saving & Loading Portfolios

1. Configure your portfolio in the sidebar
2. Enter a portfolio name
3. Click **"💾 Save Portfolio Snapshot"**
4. Load saved portfolios from the dropdown menu
5. Use **"✏️ Edit"** to rename or **"🗑️ Delete"** to remove

### Running Optimizations

1. Navigate to **"🔧 Optimizer"** tab
2. Choose optimization goal:
   - 🚀 **Max Sharpe** — best risk-adjusted returns
   - 🛡️ **Min Volatility** — lowest risk
3. Set minimum weight per asset (default: 5%)
4. Click **"Run Optimization"**
5. Review comparison table and performance charts
6. Click **"✅ Apply These Weights"** to use optimized allocation

### Generating Efficient Frontier

1. Go to **"🏔️ Efficient Frontier"** tab
2. Set number of random portfolios (500-5000)
3. Adjust your risk tolerance slider
4. Click **"🚀 Generate Efficient Frontier"**
5. Analyze your portfolio's position
6. Review optimal portfolio recommendations
7. Apply suggested weights if desired

---

## 🔧 Troubleshooting

### "No data downloaded from Yahoo Finance"

**Possible causes:**
- Invalid ticker symbols
- Date range in the future or too recent
- Yahoo Finance API rate limiting
- Network connectivity issues

**Solutions:**
1. Verify ticker format (`.NS` for NSE, `.BO` for BSE)
2. Check date range is historical
3. Try US stocks first (AAPL, MSFT) to test connection
4. Wait 1-2 minutes and retry
5. Clear browser cache


### Streamlit Cloud Deployment Issues

**Python version error:**
- Create `.python-version` file with content: `3.11`

**Module not found errors:**
- Ensure `requirements.txt` has correct versions
- Check Streamlit Cloud logs for specific missing packages

**Google Sheets authentication fails:**
- Verify all service account credentials in Streamlit Cloud secrets
- Ensure JSON formatting is correct (no extra quotes)
- Check sheet sharing permissions

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.40.2 |
| **Market Data** | yFinance 0.2.52 |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Altair 5.5.0 |
| **Optimization** | SciPy (SLSQP solver) |
| **Database** | Google Sheets (gspread) |
| **Authentication** | session_state + streamlit-cookies-manager |
| **AI Engine** | Google Gemini 2.0 Flash |
| **Deployment** | Streamlit Cloud |

---

## 🗺️ Roadmap (V2 Ideas)

Potential future enhancements:

- [ ] **Factor Analysis** — Fama-French 3/5-factor exposure
- [ ] **Risk Attribution** — MCTR (Marginal Contribution to Risk)
- [ ] **Return Attribution** — performance decomposition
- [ ] **Event Stress Testing** — COVID-19, 2022 tightening scenarios
- [ ] **PDF Export** — downloadable memo-style reports
- [ ] **Rebalancing Simulator** — periodic rebalancing strategies
- [ ] **Tax-Loss Harvesting** — optimize for after-tax returns
- [ ] **ESG Scoring** — sustainability metrics integration
- [ ] **Options Greeks** — if derivatives added
- [ ] **Backtesting Engine** — strategy performance over time

**Vote on features:** [Open an issue](https://github.com/hstg11/portfolio-analytics-dashboard-hstg/issues) with your suggestions!

---

## 🤝 Contributing

Contributions are welcome! Whether it's bug fixes, new features, or documentation improvements.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow **PEP 8** style guide
- Add **docstrings** to all functions
- **Test locally** before submitting PR
- Update **README** if adding user-facing features
- Keep commits **atomic and descriptive**

### Reporting Issues

Found a bug? [Open an issue](https://github.com/hstg11/portfolio-analytics-dashboard-hstg/issues) with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version)

---

## ⚠️ Disclaimer

This dashboard is for **educational and research purposes only**. It is **NOT** intended to provide:
- Investment advice or recommendations
- Financial planning services
- Tax or legal guidance
- Guaranteed future returns

**Important Notes:**
- All metrics are based on **historical data** and assume lump-sum investment
- **Past performance does not guarantee future results**
- Markets carry inherent risk — users are solely responsible for investment decisions
- The AI analysis feature is experimental and should not be relied upon for financial decisions
- Always consult a licensed financial advisor before making investment decisions

**Methodology Assumptions:**
- Assumes full portfolio investment at start date (no SIP/DCA)
- Does not account for transaction costs, taxes, or dividends (unless using total return data)
- Risk metrics are approximations based on historical volatility

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**You are free to:**
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

**Under the condition:**
- ⚠️ Include original license and copyright notice

---

## 👤 Author

**Harbhajan Singh Tuteja**  
Quantitative Finance & Portfolio Analytics

- 🔗 [LinkedIn](https://www.linkedin.com/in/harbhajan-singh-tuteja/)
- 💼 [GitHub](https://github.com/hstg11)
- 🌐 [Live App](https://portfolio-analytics-dashboard-hstg.streamlit.app/)

**Interested in quantitative finance projects?** Feel free to connect!

---

## 🙏 Acknowledgments

This project was made possible by these excellent open-source tools:

- **[Streamlit](https://streamlit.io/)** — for the incredible web framework
- **[yFinance](https://github.com/ranaroussi/yfinance)** — for free market data access
- **[SciPy](https://scipy.org/)** — for optimization algorithms
- **[Altair](https://altair-viz.github.io/)** — for declarative visualizations
- **[Google Sheets API](https://developers.google.com/sheets/api)** — for data persistence
- **[Gemini API](https://ai.google.dev/)** — for AI-powered insights

Special thanks to the open-source community for making projects like this possible.

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=hstg11/portfolio-analytics-dashboard-hstg&type=Date)](https://star-history.com/#hstg11/portfolio-analytics-dashboard-hstg&Date)

---

<div align="center">

**Built with ❤️ for the quantitative finance community**



</div>