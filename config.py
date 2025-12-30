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
