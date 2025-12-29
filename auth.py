"""
Authentication and user management module
Handles login, signup, and user data storage with validation
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from streamlit_cookies_manager import EncryptedCookieManager
from google.oauth2.service_account import Credentials
import gspread

from config import (
    COOKIE_PREFIX, COOKIE_PASSWORD, GOOGLE_SHEETS_SCOPE,
    SHEET_NAME, USERS_WORKSHEET, PORTFOLIOS_WORKSHEET
)


# ============================================
# VALIDATION FUNCTIONS
# ============================================

def validate_email(email):
    """Validate email format"""
    if not email:
        return False, "Email is required."
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Please enter a valid email address (e.g., user@example.com)."
    
    return True, ""


def validate_phone(phone):
    """Validate phone number (must be exactly 10 digits)"""
    if not phone:
        return False, "Phone number is required."
    
    cleaned = re.sub(r'[\s\-\(\)]', '', phone)
    
    if not cleaned.isdigit():
        return False, "Phone number must contain only digits."
    
    if len(cleaned) != 10:
        return False, "Phone number must be exactly 10 digits."
    
    return True, cleaned


def validate_name(name):
    """Validate name (not empty, reasonable length)"""
    if not name:
        return False, "Name is required."
    
    if len(name.strip()) < 2:
        return False, "Name must be at least 2 characters."
    
    if len(name) > 100:
        return False, "Name is too long (max 100 characters)."
    
    return True, ""


def validate_city(city):
    """Validate city (not empty)"""
    if not city:
        return False, "City is required."
    
    if len(city.strip()) < 2:
        return False, "City must be at least 2 characters."
    
    return True, ""


# ============================================
# COOKIE MANAGER
# ============================================

def init_cookie_manager():
    """Initialize and return cookie manager"""
    cookies = EncryptedCookieManager(
        prefix=COOKIE_PREFIX,
        password=COOKIE_PASSWORD
    )
    
    if not cookies.ready():
        st.stop()
    
    return cookies


# ============================================
# GOOGLE SHEETS CONNECTION
# ============================================

@st.cache_resource
def init_google_sheets():
    """Initialize Google Sheets connection (cached)"""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=GOOGLE_SHEETS_SCOPE
    )
    
    gc = gspread.authorize(creds)
    sh = gc.open(SHEET_NAME)
    
    users_ws = sh.worksheet(USERS_WORKSHEET)
    portfolios_ws = sh.worksheet(PORTFOLIOS_WORKSHEET)
    
    return users_ws, portfolios_ws


@st.cache_data(ttl=600)
def fetch_users_data(_users_ws):
    """Fetch users data from Google Sheets (cached for 10 min)"""
    try:
        data = _users_ws.get_all_records()
        df = pd.DataFrame(data)
        if not df.empty:
            df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return pd.DataFrame()


# ============================================
# HELPER FUNCTIONS
# ============================================

def now_ist():
    """Return current datetime in IST"""
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")


def user_exists(email, users_df):
    """Check if user exists in database"""
    if users_df.empty:
        return False
    if "email" not in users_df.columns:
        return False
    return email.lower() in users_df["email"].str.lower().values


def save_user(email, name, phone, city, users_ws):
    """Save or update user in Google Sheets"""
    now = now_ist()
    
    try:
        records = users_ws.col_values(1)
        
        if email in records:
            row_index = records.index(email) + 1
            users_ws.update_cell(row_index, 6, now)
        else:
            users_ws.append_row([email, name, phone, city, now, now])
    except Exception as e:
        st.error(f"Error saving user: {e}")


def save_portfolio(email, name, portfolio_name, tickers, weights, portfolios_ws, opt_weights=None):
    """Save portfolio snapshot to Google Sheets with sequential ID"""
    try:
        all_records = portfolios_ws.get_all_records()
        
        user_portfolios = [
            row for row in all_records 
            if str(row.get('email')).strip().lower() == email.strip().lower()
        ]
        
        next_index = len(user_portfolios) + 1
        portfolio_id = f"{email}_{next_index}"

        tickers_str = ",".join(tickers)
        weights_str = ",".join([str(round(float(w), 4)) for w in weights])
        
        opt_weights_str = ""
        if opt_weights is not None:
            opt_weights_str = ",".join([str(round(float(w), 4)) for w in opt_weights])

        portfolios_ws.append_row([
            email,
            name,
            portfolio_id,
            portfolio_name,
            tickers_str,
            weights_str,
            opt_weights_str,
            now_ist()
        ])
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {e}")
        return False


# ============================================
# SIGNUP GATE UI
# ============================================

def show_signup_gate(users_df, users_ws, cookies):
    """Display signup/login screen with validation"""
    
    # Check for auto-login cookie FIRST
    if "last_seen_updated" not in st.session_state:
        st.session_state["last_seen_updated"] = False
    
    stored_email = cookies.get("user_email")
    if stored_email and not st.session_state.get("signed_up"):
        # Auto-login
        st.session_state["signed_up"] = True
        st.session_state["user_email"] = stored_email
        
        # Update last_seen only once
        if not st.session_state["last_seen_updated"]:
            if user_exists(stored_email, users_df):
                user_row = users_df[users_df["email"] == stored_email.lower()]
                if not user_row.empty:
                    name = user_row.iloc[0].get("name", "")
                    phone = user_row.iloc[0].get("phone", "")
                    city = user_row.iloc[0].get("city", "")
                    save_user(stored_email, name, phone, city, users_ws)
            st.session_state["last_seen_updated"] = True
        
        st.rerun()
    
    # Show signup form
    st.title("🔐 Welcome – Please Sign Up")
    st.write("Please fill in your details to access the portfolio dashboard.")
    
    with st.form("signup_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Full Name *",
                placeholder="e.g., John Doe",
                help="Your full name"
            )
        
        with col2:
            email = st.text_input(
                "Email Address *",
                placeholder="e.g., john@example.com",
                help="We'll use this to identify you"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            phone = st.text_input(
                "Mobile Number *",
                placeholder="e.g., 9876543210",
                help="10-digit mobile number",
                max_chars=10
            )
        
        with col4:
            city = st.text_input(
                "City *",
                placeholder="e.g., Mumbai",
                help="Your current city"
            )
        
        st.caption("* Required fields")
        
        submitted = st.form_submit_button("Continue ➜", type="primary", use_container_width=True)
    
    if submitted:
        errors = []
        
        name_valid, name_error = validate_name(name)
        if not name_valid:
            errors.append(name_error)
        
        email_valid, email_error = validate_email(email)
        if not email_valid:
            errors.append(email_error)
        
        phone_valid, phone_result = validate_phone(phone)
        if not phone_valid:
            errors.append(phone_result)
        else:
            phone_cleaned = phone_result
        
        city_valid, city_error = validate_city(city)
        if not city_valid:
            errors.append(city_error)
        
        if errors:
            for error in errors:
                st.error(error)
            st.stop()
        
        # Check if returning user
        if user_exists(email, users_df):
            st.success("✅ Welcome back! Loading dashboard…")
            st.session_state["signed_up"] = True
            st.session_state["user_email"] = email
            cookies["user_email"] = email
            cookies.save()
            save_user(email, name, phone_cleaned, city, users_ws)
            st.rerun()
        else:
            st.success("✅ Signup complete! Loading dashboard…")
            save_user(email, name, phone_cleaned, city, users_ws)
            st.session_state["signed_up"] = True
            st.session_state["user_email"] = email
            cookies["user_email"] = email
            cookies.save()
            st.rerun()
    
    st.stop()  # ✅ CRITICAL: Blocks dashboard rendering


# ============================================
# LOGOUT
# ============================================

def logout(cookies):
    """Handle user logout"""
    if "user_email" in cookies:
        cookies.delete("user_email")
        cookies.save()
    
    st.session_state["signed_up"] = False
    st.session_state["user_email"] = None
    st.session_state["last_seen_updated"] = False
    
    st.rerun()