# config.py
import streamlit as st

# ------------------ Cookie Settings ------------------
COOKIE_PREFIX = "portfolio_dashboard"
# config.py

COOKIE_PASSWORD = st.secrets.get("COOKIE_PASSWORD", "fallback_only_for_local")
# ------------------ Google Sheets ------------------
GOOGLE_SHEETS_SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "portfolio_users"
USERS_WORKSHEET = "users"
PORTFOLIOS_WORKSHEET = "portfolios"
