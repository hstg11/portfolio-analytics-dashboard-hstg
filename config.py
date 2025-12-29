import streamlit as st
from google.oauth2.service_account import Credentials
import gspread

# Constants
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
SHEET_NAME = "portfolio_users"

@st.cache_resource
def init_google_sheets():
    """Authorize and open Google Sheets, return users and portfolios worksheets."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPE
    )
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    return sh.worksheet("users"), sh.worksheet("portfolios")

# config.py

# ------------------ Cookie Settings ------------------
COOKIE_PREFIX = "portfolio_dashboard"
COOKIE_PASSWORD = "super_secret_password_change_me"

# ------------------ Google Sheets ------------------
GOOGLE_SHEETS_SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SHEET_NAME = "portfolio_users"
USERS_WORKSHEET = "users"
PORTFOLIOS_WORKSHEET = "portfolios"
    

