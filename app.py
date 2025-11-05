import streamlit as st
import pandas as pd
import json
from pathlib import Path
import hashlib # For password hashing
# Keep DB imports ONLY IF pipeline still uses them for something else (like saving news)
# If not using MongoDB at all, you can remove these:
# from src.config import db_available, db, OPENAI_API_KEY
# Assume db_available might still be useful if news saving uses DB
from src.config import db_available, db

from src.data_collection import get_stock_data, scrape_multiple_news
# Ensure pipeline is imported if used
from src.nlp_sentiment_predictor import sentiment_pipeline as nlp_sentiment_pipeline
from datetime import datetime, timedelta
# Import Gemini if used
import google.generativeai as genai
import os
import plotly.graph_objects as go
import yfinance as yf
import pytz # Keep if used for timestamps
import numpy as np # Keep if used
from dotenv import load_dotenv # Import dotenv

# Load .env file (should happen early)
load_dotenv()

# --- CONFIGURATION & SETUP ---

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load from .env
gemini_model = None
GEMINI_AVAILABLE = False
if GEMINI_API_KEY:
    try:
        # No need to re-import if already done at top
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash") # Use specific model like flash
        GEMINI_AVAILABLE = True # Set flag to True on success
        print("✅ Gemini Configured Successfully")
    except ImportError:
        # st.error("❌ Failed to import google.generativeai. Install it (`pip install google-generativeai`)")
        print("--- ERROR --- Failed to import google.generativeai") # Print error for backend
    except Exception as e:
        # st.error(f"❌ Error configuring Gemini: {e}") # Show error in UI if needed
        print(f"--- ERROR --- Error configuring Gemini: {e}") # Log the error
        # GEMINI_AVAILABLE remains False, gemini_model remains None
else:
    print("⚠️ GEMINI_API_KEY not set.") # Log this


# --- JSON User Database ---
USER_DB_PATH = Path("users.json") # Define path for user credentials and watchlists

# Helper Functions for User Management
def load_users():
    """Loads user data from the JSON file."""
    if USER_DB_PATH.exists():
        try:
            with open(USER_DB_PATH, "r") as f:
                content = f.read()
                if not content: return {} # Handle empty file gracefully
                return json.loads(content)
        except json.JSONDecodeError:
             st.error("Error reading user data file (corrupted?). Starting fresh or correct the file.")
             return {} # Return empty to allow potential registration
        except Exception as e:
             st.error(f"Error loading users: {e}")
             return {}
    return {} # Return empty if file doesn't exist

def save_users(users_data):
    """Saves user data dictionary to the JSON file."""
    try:
        with open(USER_DB_PATH, "w") as f:
            json.dump(users_data, f, indent=4)
        print("User data saved successfully.") # Confirmation log
        return True
    except Exception as e:
         st.error(f"Failed to save user data: {e}")
         return False

def hash_password(password):
    """Hashes a password using SHA256. Simple, consider bcrypt for production."""
    if not password: raise ValueError("Password cannot be empty")
    return hashlib.sha256(password.encode()).hexdigest()

def check_password_hash(stored_hash, provided_password):
    """Checks if the provided password matches the stored SHA256 hash."""
    if not stored_hash or not provided_password:
        return False
    return stored_hash == hash_password(provided_password)

def authenticate_user(username, password):
    """Checks credentials against the stored hashed passwords."""
    users = load_users()
    # Check if username exists and password matches
    if username in users and check_password_hash(users[username].get('password'), password):
        # Set session state variables on successful login
        st.session_state['authentication_status'] = True
        st.session_state['name'] = username # Use username as the 'name' identifier
        st.session_state['watchlist'] = users[username].get('watchlist', []) # Load watchlist
        st.session_state['menu_selection'] = 'App' # Set state to show main app
        print(f"Authentication successful for {username}") # Debug print
        return True
    else:
        print(f"Authentication failed for {username}") # Debug print
        st.session_state['authentication_status'] = False
        return False

def logout_user():
    """Clears relevant session state variables and resets page state for login."""
    print("Logging out user...") # Debug print
    # List of keys specific to user session or analysis results
    keys_to_clear = [
        'authentication_status', 'name', 'watchlist', 'menu_selection',
        'stock_data', 'stock_info', 'prediction', 'prediction_fig',
        'analyzed_df', 'gemini_summary', 'summary_ticker',
        'loaded_ticker', 'selected_ticker', 'analysis_ticker',
        'run_analysis_on_select', 'load_data_on_select' # Clear flags too
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    # Explicitly set status to None and page to Login after clearing
    st.session_state['authentication_status'] = None
    st.session_state['menu_selection'] = 'Login'
    st.rerun() # Force rerun to display login page

# --- Initialize Session State ---
# Ensures keys exist on the very first run of the session
if 'authentication_status' not in st.session_state: st.session_state['authentication_status'] = None
if 'name' not in st.session_state: st.session_state['name'] = None
if 'menu_selection' not in st.session_state: st.session_state['menu_selection'] = 'Login'
if 'watchlist' not in st.session_state: st.session_state['watchlist'] = []
if 'selected_ticker' not in st.session_state: st.session_state.selected_ticker = ""


# --- Page Configuration (Set Once) ---
st.set_page_config(
    page_title='SynapseStreet - AI Stock Predictor',
    layout='wide',
    initial_sidebar_state='auto', # Let Streamlit decide based on width, or use 'expanded'
    page_icon='🧠'
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* ... Your existing CSS rules ... */
    .main-header { font-size: 3rem; font-weight: 700; background: linear-gradient(120deg, #2193b0, #6dd5ed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# --- LOGIN AND REGISTRATION PAGE FUNCTIONS ---

def registration_page():
    st.markdown('<h1 class="main-header" style="text-align: center;">🧠 Register</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1]) # Centering
    with col2:
        with st.form("Registration_form"):
            st.subheader("Create New Account") # Added subheader
            new_username = st.text_input("Choose Username", max_chars=20) # Slightly longer max chars
            new_password = st.text_input("Choose Password (min 6 chars)", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register", type="primary", use_container_width=True)

            if submitted:
                users = load_users() # Load current users
                # Perform all validations
                if not new_username: st.error("Username cannot be empty.")
                elif not new_password: st.error("Password cannot be empty.")
                elif new_username in users: st.error("Username already taken.")
                elif new_password != confirm_password: st.error("Passwords do not match.")
                elif len(new_password) < 6: st.error("Password must be at least 6 characters.")
                else:
                    # If all checks pass, add user
                    users[new_username] = {
                        'password': hash_password(new_password),
                        'watchlist': ['AAPL', 'GOOGL', 'TSLA'] # Sensible default watchlist
                    }
                    if save_users(users): # Attempt to save
                        st.success("✅ Registration successful! Please log in.")
                        st.session_state['menu_selection'] = 'Login'
                        st.rerun() # Go to login page
                    # else: Error message shown by save_users()

        st.divider()
        if st.button("Go back to Login", use_container_width=True):
            st.session_state['menu_selection'] = 'Login'
            st.rerun()

def login_page():
    st.markdown('<h1 class="main-header" style="text-align: center;">🧠 Login</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1]) # Centering
    with col2:
        # Display error message from previous failed attempt if exists
        if st.session_state.get('authentication_status') == False:
            st.error("Invalid username or password.")
            st.session_state.authentication_status = None # Reset status after showing error

        with st.form("Login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                print(f"--- DEBUG --- Login attempt for: {username}")
                if authenticate_user(username, password):
                    print(f"--- DEBUG --- Login success for {username}. Rerunning.")
                    # State changes handled in authenticate_user()
                    st.rerun() # Redirects to 'App' state defined in authenticate_user
                # else: authenticate_user sets status to False, error shown on next rerun

        st.divider()
        if st.button("Register New User", use_container_width=True):
            st.session_state['menu_selection'] = 'Register'
            st.rerun()

        # Suggest registration if user file doesn't exist or is empty
        if not USER_DB_PATH.exists() or not load_users():
             st.info("No users found. Please register an account.")


# --- MAIN APPLICATION ROUTING ---

# Check authentication status first (using .get for safety)
# If True, render the main app. Otherwise, render login/register.
if not st.session_state.get('authentication_status'):
    # Show Login or Register page based on menu_selection state
    if st.session_state.get('menu_selection') == 'Register':
        registration_page()
    else: # Default to Login
        login_page()
    # No st.stop() needed here, execution ends naturally if not authenticated

else: # --- IF LOGGED IN, CONTINUE TO MAIN APP ---
    # (User is authenticated if this point is reached)
    print(f"--- DEBUG --- Rendering Main App for user: {st.session_state.get('name')}")

    # --- Predefined Tickers ---
    POPULAR_TICKERS = [ # Keep your original list here
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM',
        'TSLA', 'AMD', 'INTC', 'CSCO', 'QCOM', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP',
        'JNJ', 'PFE', 'LLY', 'UNH', 'MRK', 'ABBV', 'TMO',
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'LOW', 'NKE', 'MCD',
        'BA', 'CAT', 'GE', 'LMT', 'XOM', 'CVX', 'DUK', 'NEE',
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS',
        'RELIANCE.NS', 'LT.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'ULTRACEMCO.NS',
        'SUNPHARMA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS',
        'HUL.NS', 'ITC.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'BHARTIARTL.NS', 'TITAN.NS', 'ADANIENT.NS'
    ]

    # --- Sidebar Configuration (Logged In) ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.header('🎛 Controls')
        st.success(f"👋 Logged in as *{st.session_state.get('name')}*")
        st.button('🏃 Logout', on_click=logout_user, use_container_width=True)
        st.divider()

        # --- Watchlist Display & Edit ---
        st.subheader("My Watchlist")
        user_watchlist = st.session_state.get('watchlist', []) # Get from session state
        if not user_watchlist:
            st.info("Your watchlist is empty.")
        else:
            st.write("Click ticker to analyze:") # Instruction
            cols = st.columns(3) # Adjust column count as needed
            col_idx = 0
            for item in sorted(user_watchlist):
                with cols[col_idx % 3]:
                    if st.button(item, key=f"watch_{item}", use_container_width=True):
                        print(f"--- DEBUG --- Watchlist button clicked: {item}")
                        st.session_state.selected_ticker = item # Update selected ticker
                        # --- ⚙️ FLAG TO RUN ANALYSIS ON SELECT ---
                        st.session_state.run_analysis_on_select = True
                        # --- ⚙️ FLAG TO ENSURE DATA LOADS ---
                        st.session_state.load_data_on_select = True
                        # Clear previous analysis data
                        keys_to_clear = ['stock_data', 'stock_info', 'prediction', 'prediction_fig', 'analyzed_df', 'gemini_summary', 'summary_ticker', 'loaded_ticker', 'analysis_ticker']
                        for key in keys_to_clear:
                            if key in st.session_state: del st.session_state[key]
                        st.rerun()
                col_idx += 1

        with st.expander("✏️ Edit Watchlist"):
            add_ticker_input = st.text_input("Add Ticker", key="add_ticker").strip().upper()
            if st.button("Add"): # Add Button Logic
                if add_ticker_input and add_ticker_input not in user_watchlist:
                    try: # Validate
                        if not yf.Ticker(add_ticker_input).history(period='1d').empty:
                            user_watchlist.append(add_ticker_input)
                            users_data = load_users()
                            # Ensure user exists in data (should always be true if logged in)
                            if st.session_state['name'] in users_data:
                                 users_data[st.session_state['name']]['watchlist'] = sorted(user_watchlist)
                                 if save_users(users_data):
                                     st.session_state['watchlist'] = sorted(user_watchlist) # Update state
                                     st.success(f"`{add_ticker_input}` added.")
                                     st.rerun()
                                 # else: Error handled in save_users
                            else: st.error("Current user data error.") # Data consistency issue
                        else: st.warning(f"`{add_ticker_input}` invalid.")
                    except Exception as e: st.warning(f"Validation error: {e}")
                elif not add_ticker_input: st.warning("Enter ticker.")
                else: st.warning("Already in watchlist.")

            remove_ticker_select = st.selectbox("Remove Ticker", [""] + sorted(user_watchlist), key="remove_ticker")
            if st.button("Remove"): # Remove Button Logic
                if remove_ticker_select and remove_ticker_select in user_watchlist:
                    user_watchlist.remove(remove_ticker_select)
                    users_data = load_users()
                    if st.session_state['name'] in users_data:
                        users_data[st.session_state['name']]['watchlist'] = sorted(user_watchlist)
                        if save_users(users_data):
                            st.session_state['watchlist'] = sorted(user_watchlist) # Update state
                            st.success(f"`{remove_ticker_select}` removed.")
                            st.rerun()
                        # else: Error handled in save_users
                    else: st.error("Current user data error.")
                elif not remove_ticker_select: st.warning("Select ticker.")
        # --- End Watchlist ---

        st.divider()
        # Combine popular and user watchlist for the main dropdown
        available_tickers = sorted(list(set(POPULAR_TICKERS + user_watchlist)))

        # Use session state selected_ticker for default value
        try: default_index = ([''] + available_tickers).index(st.session_state.get('selected_ticker', ''))
        except ValueError: default_index = 0

        # Main Ticker Selector (in sidebar)
        ticker_sb = st.selectbox(
            '📊 Select/Search Stock',
            options=[''] + available_tickers,
            index=default_index, # Set based on session state
            placeholder="Type or select...",
            help="Select or type a stock ticker symbol",
            key="main_ticker_selector_sidebar"
        ).strip().upper()

        # Update session state if sidebar selector changes
        if ticker_sb != st.session_state.selected_ticker:
             st.session_state.selected_ticker = ticker_sb
             # Clear analysis state when ticker changes
             keys_to_clear = ['stock_data', 'stock_info', 'prediction', 'prediction_fig', 'analyzed_df', 'gemini_summary', 'summary_ticker', 'loaded_ticker', 'analysis_ticker']
             for key in keys_to_clear:
                 if key in st.session_state: del st.session_state[key]
             # --- ⚙️ Reset auto-run flag if main dropdown used ---
             st.session_state.run_analysis_on_select = False
             st.session_state.load_data_on_select = False # Reset auto-load flag
             st.rerun() # Rerun immediately

        # Final ticker assignment from session state
        ticker = st.session_state.selected_ticker

        st.divider()
        # Analysis Options
        st.subheader("Analysis Options")
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30, 1, help="Prediction length")
        lookback_period = st.selectbox("Historical Period", ['3mo', '6mo', '1y', '2y', '5y'], 1, help="Chart & AI data (Min 6mo for AI)") # Index 1 = 6mo

        st.divider()
        # Action Buttons
        run_analysis_button = st.button('🚀 Run Full Analysis', type='primary', use_container_width=True)
        refresh_data_button = st.button('🔄 Refresh Hist. Data', use_container_width=True)

    # --- Main Content Area (Logged In) ---
    st.markdown('<h1 class="main-header">🧠 SynapseStreet</h1>', unsafe_allow_html=True) # Show title in main area too
    st.markdown("*AI-Powered Stock Analysis combining Sentiment & Machine Learning*")

    if not ticker:
        # Landing Page (Logged In, No Ticker Selected)
        st.info("👈 Select a stock ticker from the sidebar.")
        # ... (Quick Access Buttons) ...
        st.divider(); st.subheader("🔥 Quick Access")
        quick_cols = st.columns(6); quick_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
        for idx, qt in enumerate(quick_tickers):
            with quick_cols[idx]:
                if st.button(qt, key=f"quick_{qt}", use_container_width=True):
                    st.session_state.selected_ticker = qt
                    st.session_state.run_analysis_on_select = True # Also auto-run
                    st.session_state.load_data_on_select = True # Also auto-load
                    # Clear previous analysis data
                    keys_to_clear_quick = ['stock_data', 'stock_info', 'prediction', 'prediction_fig', 'analyzed_df', 'gemini_summary', 'summary_ticker', 'loaded_ticker', 'analysis_ticker']
                    for key in keys_to_clear_quick:
                        if key in st.session_state: del st.session_state[key]
                    st.rerun()

    else:
        # --- Main Analysis Interface ---
        st.markdown(f"## Analysis for *{ticker}*")
        currency_symbol = "₹" if ticker.endswith(".NS") else "$"

        tab_titles = ["📊 Historical", "🔮 AI Predict", "📰 News/Tally", "📈 Technicals", "🧩 AI Summary", "🏢 Profile"]
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

        # Helper function
        def get_start_date_from_period(period_str):
            today=datetime.now(); days_map = {'3mo':90,'6mo':182,'1y':365,'2y':730,'5y':1825}
            delta = timedelta(days=days_map.get(period_str, 365)); return today - delta

        # --- TAB 1: Historical Data ---
        with tab1:
            st.subheader(f"Historical Stock Data - {ticker}")
            # --- Data Loading Logic ---
            load_trigger = refresh_data_button or ('stock_data' not in st.session_state or st.session_state.get('loaded_ticker') != ticker) or st.session_state.get('load_data_on_select', False)
            if run_analysis_button and ('stock_data' not in st.session_state or st.session_state.get('loaded_ticker') != ticker): load_trigger = True # Force load if analysis clicked and no data

            print(f"--- DEBUG (Tab 1) --- Load Trigger: {load_trigger}, Refresh: {refresh_data_button}, Load Flag: {st.session_state.get('load_data_on_select', False)}, Ticker Match: {st.session_state.get('loaded_ticker') == ticker}")

            if load_trigger:
                try:
                    # Reset load flag *before* loading
                    st.session_state.load_data_on_select = False
                    print("--- DEBUG (Tab 1) --- Reset load_data_on_select flag.")

                    start_date = get_start_date_from_period(lookback_period); start_date_str = start_date.strftime('%Y-%m-%d')
                    with st.spinner(f'Fetching data since {start_date_str}...'): df = get_stock_data(ticker, start=start_date_str)
                    if df is None or df.empty: st.error(f'No data for *{ticker}*.'); df = None
                    else: st.success(f'✅ Fetched {len(df)} rows.')
                    st.session_state.stock_data = df; st.session_state.loaded_ticker = ticker
                    if df is not None:
                         with st.spinner("Fetching profile..."): stock_info = yf.Ticker(ticker).info; st.session_state.stock_info = stock_info
                    else:
                         if 'stock_info' in st.session_state: del st.session_state.stock_info
                    # Clear analysis state ONLY if refresh button pressed
                    if refresh_data_button:
                         keys_to_clear_analysis = ['prediction', 'prediction_fig', 'analyzed_df', 'gemini_summary', 'summary_ticker', 'analysis_ticker']
                         for key in keys_to_clear_analysis:
                             if key in st.session_state: del st.session_state[key]
                    st.rerun() # Rerun needed after loading
                except Exception as e: st.error(f'❌ Fetch error: {e}'); st.exception(e); st.session_state.stock_data = None; st.session_state.loaded_ticker = ticker; st.session_state.load_data_on_select = False;

            # --- Display Data ---
            if 'stock_data' in st.session_state and st.session_state.stock_data is not None and st.session_state.loaded_ticker == ticker:
                df = st.session_state.stock_data
                # ... (Display Metrics, Table, Chart Toggle - same as before) ...
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.to_flat_index(); df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                col1, col2, col3, col4 = st.columns(4)
                with col1: current_price = df['Close'].iloc[-1]; st.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
                if len(df) > 1:
                    with col2: day_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]; day_change_pct = (day_change / df['Close'].iloc[-2]) * 100 if df['Close'].iloc[-2] !=0 else 0; st.metric("Day Change", f"{currency_symbol}{day_change:.2f}", f"{day_change_pct:+.2f}%")
                else:
                    with col2: st.metric("Day Change", "N/A")
                    with col3: period_high = df['High'].max(); st.metric(f"Period High ({lookback_period})", f"{currency_symbol}{period_high:.2f}")
                    with col4: avg_volume = df['Volume'].tail(20).mean(); st.metric("Avg Vol (20D)", f"{avg_volume/1e6:.2f}M" if avg_volume else "N/A")
                st.divider(); st.subheader(f"Data Table ({lookback_period})"); st.dataframe(df, use_container_width=True, height=400)
                st.divider(); col1c, col2c = st.columns([3, 1]);
                with col1c: st.subheader(f"Price History ({lookback_period})")
                with col2c: chart_type_tab1 = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True, key="chart_toggle_tab1", label_visibility="collapsed")
                if chart_type_tab1 == "Line": st.line_chart(df['Close'], use_container_width=True)
                else: fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker)]); fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title=f"Price ({currency_symbol})"); st.plotly_chart(fig, use_container_width=True)

            elif 'stock_data' in st.session_state and st.session_state.stock_data is None and st.session_state.loaded_ticker == ticker: st.error(f"Failed to load data for {ticker}.")
            else: st.info("Load historical data using sidebar button.")


        # --- TAB 2: AI Predictions ---
        with tab2:
            st.subheader(f"🔮 AI Predictions - {ticker}")
            # --- Trigger Logic ---
            should_run_analysis = run_analysis_button or st.session_state.get('run_analysis_on_select', False)
            print(f"--- DEBUG (Tab 2) --- Should Run: {should_run_analysis}, Auto Flag: {st.session_state.get('run_analysis_on_select', False)}, Button: {run_analysis_button}")

            if should_run_analysis:
                st.session_state.run_analysis_on_select = False # Reset flag
                print("--- DEBUG (Tab 2) --- Auto-run flag reset.")

                if lookback_period in ['1mo', '3mo']: st.error("❌ Min '6mo' required for AI.")
                elif 'stock_data' not in st.session_state or st.session_state.stock_data is None or st.session_state.get('loaded_ticker') != ticker:
                     if should_run_analysis and not run_analysis_button: # Auto-trigger specific message
                         st.warning("⚠️ Data needs loading first. Click 'Load Historical Data' in Tab 1 or sidebar button.")
                     else: # Manual button press message
                         st.warning("⚠️ Load/refresh valid historical data first (Tab 1 or sidebar).")
                else: # Data ready, run analysis
                    try:
                        with st.spinner(f'🧠 Running AI analysis...'): analyzed_df, prediction, fig = nlp_sentiment_pipeline(ticker, forecast_days=forecast_days, lookback_period=lookback_period)
                        st.session_state.prediction = prediction; st.session_state.prediction_fig = fig; st.session_state.analyzed_df = analyzed_df; st.session_state.analysis_ticker = ticker
                        print(f"--- DEBUG (Tab 2) --- Analysis SUCCESS. Stored results for ticker: {ticker}")
                        if fig is None: st.error(f"Pred failed: {prediction.get('error', 'Unknown')}")
                        else: st.success("✅ Pred ready!"); st.rerun() # Rerun to display
                    except Exception as e: 
                        st.error(f"❌ Pred error: {e}"); st.exception(e); # ... (clear state on error) ...
                        keys_to_clear_fail = ['prediction', 'prediction_fig', 'analyzed_df']
                        for key in keys_to_clear_fail:
                            if key in st.session_state: del st.session_state[key]
                        st.session_state.analysis_ticker = ticker

            # --- Display Logic ---
            if 'prediction_fig' in st.session_state and st.session_state.prediction_fig is not None and 'prediction' in st.session_state and st.session_state.get('analysis_ticker') == ticker:
                prediction = st.session_state.prediction; fig = st.session_state.prediction_fig
                # ... (Display metrics, confidence, chart, details, save button - same as before) ...
                col1, col2, col3, col4 = st.columns(4);
                with col1: st.metric("Trend", prediction.get('trend', 'N/A'))
                with col2: st.metric("Sentiment", f"{prediction.get('avg_sentiment', 0):+.3f}")
                with col3: st.metric("7D Target", prediction.get('target_7d', 'N/A'), prediction.get('predicted_change_7d', 'N/A'))
                with col4: st.metric("30D Target", prediction.get('target_30d', 'N/A'), prediction.get('predicted_change_30d', 'N/A'))
                st.divider(); confidence = prediction.get('confidence', 'Unknown');
                if confidence == 'High': st.success(f"🎯 Confidence: **{confidence}**")
                elif confidence == 'Medium': st.info(f"📊 Confidence: **{confidence}**")
                else: st.warning(f"⚠️ Confidence: **{confidence}**")
                st.divider(); st.plotly_chart(fig, use_container_width=True)
                with st.expander("📋 Details"): st.json(prediction)
                if st.button("💾 Save Chart", key="save_chart_btn"):
                     if fig: os.makedirs("predictions", exist_ok=True); fpath = f"predictions/{ticker}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"; fig.write_html(fpath); st.success(f"✅ Saved to {fpath}")

            elif 'prediction' in st.session_state and st.session_state.prediction and st.session_state.prediction.get('error') and st.session_state.get('analysis_ticker') == ticker: st.error(f"Pred failed: {st.session_state.prediction.get('error')}")
            else: st.info("Click 'Run Full Analysis' or select from Watchlist to run analysis.")


        # --- Tabs 3-6 (Ensure checks for session state data and ticker match) ---
        with tab3: # --- TAB 3: News & Sentiment ---
            st.subheader(f"📰 News Sentiment Analysis - {ticker}")
            def highlight_sentiment(row):
                sv=row.get('sentiment_value',0);color='#080808'; # ... (highlight logic) ...
                if sv > 0: color='#008B8B';
                elif sv < 0: color='#E44D2E';
                return [f'background-color: {color}'] * len(row)
            if 'analyzed_df' in st.session_state and st.session_state.get('analysis_ticker') == ticker:
                 analyzed_df_display = st.session_state.analyzed_df
                 if analyzed_df_display is not None and not analyzed_df_display.empty: # ... (Display distribution and details) ...
                     st.divider(); st.subheader("Distribution"); sentiment_counts = analyzed_df_display['sentiment'].value_counts()
                     col1, col2, col3 = st.columns(3);
                     with col1: positive = sentiment_counts.get('positive', 0) + sentiment_counts.get('POSITIVE', 0); st.metric("🟢 Pos", positive)
                     with col2: neutral = sentiment_counts.get('neutral', 0) + sentiment_counts.get('NEUTRAL', 0); st.metric("🟡 Neu", neutral)
                     with col3: negative = sentiment_counts.get('negative', 0) + sentiment_counts.get('NEGATIVE', 0); st.metric("🔴 Neg", negative)
                     st.divider(); st.subheader("Detailed Breakdown")
                     display_cols_news = ['timestamp', 'title', 'sentiment', 'score', 'sentiment_value', 'source']; display_df_news = analyzed_df_display[[col for col in display_cols_news if col in analyzed_df_display.columns]].copy()
                     if 'timestamp' in display_df_news.columns: display_df_news['timestamp'] = pd.to_datetime(display_df_news['timestamp']).dt.strftime('%Y-%m-%d %H:%M'); display_df_news.rename(columns={'timestamp': 'Date/Time'}, inplace=True)
                     st.dataframe(display_df_news.style.apply(highlight_sentiment, axis=1), use_container_width=True, height=400, hide_index=True)
                 elif analyzed_df_display is not None and analyzed_df_display.empty : st.info("No news found last run.")
            else: st.info("Click 'Run Full Analysis'.")

        with tab4: # --- TAB 4: Technical Indicators ---
             st.subheader(f"📈 Technical Indicators - {ticker}")
             if 'stock_data' in st.session_state and st.session_state.stock_data is not None and st.session_state.loaded_ticker == ticker:
                 df_tech = st.session_state.stock_data.copy()
                 if not all(col in df_tech.columns for col in ['Close', 'Open', 'High', 'Low']): st.warning("Missing OHLC columns.")
                 else:
                     try: # ... (Calculate MAs) ...
                         df_tech['MA_20'] = df_tech['Close'].rolling(window=20, min_periods=1).mean(); df_tech['MA_50'] = df_tech['Close'].rolling(window=50, min_periods=1).mean(); df_tech['MA_200'] = df_tech['Close'].rolling(window=200, min_periods=1).mean()
                     except Exception as e_ma: st.warning(f"MA calc error: {e_ma}"); df_tech['MA_20']=None; df_tech['MA_50']=None; df_tech['MA_200']=None
                     chart_col1_t4, chart_col2_t4 = st.columns([3, 1]); # ... (Chart toggle) ...
                     with chart_col1_t4: st.subheader("Price vs. MAs")
                     with chart_col2_t4: chart_type_tab4 = st.radio("Chart Type", ["Line", "Candle"], horizontal=True, key="chart_toggle_tab4", label_visibility="collapsed")
                     chart_df_tech = df_tech.tail(252)
                     if chart_type_tab4 == "Line": st.line_chart(chart_df_tech[['Close', 'MA_20', 'MA_50', 'MA_200']], use_container_width=True)
                     else: # ... (Candlestick + MAs plot) ...
                         fig_tech = go.Figure();
                         fig_tech.add_trace(go.Candlestick(x=chart_df_tech.index, open=chart_df_tech['Open'], high=chart_df_tech['High'], low=chart_df_tech['Low'], close=chart_df_tech['Close'], name="Price"));
                         if 'MA_20' in chart_df_tech.columns and chart_df_tech['MA_20'].notna().any(): fig_tech.add_trace(go.Scatter(x=chart_df_tech.index, y=chart_df_tech['MA_20'], mode='lines', name='MA 20', line=dict(color='yellow', width=1)));
                         if 'MA_50' in chart_df_tech.columns and chart_df_tech['MA_50'].notna().any(): fig_tech.add_trace(go.Scatter(x=chart_df_tech.index, y=chart_df_tech['MA_50'], mode='lines', name='MA 50', line=dict(color='orange', width=1)));
                         if 'MA_200' in chart_df_tech.columns and chart_df_tech['MA_200'].notna().any(): fig_tech.add_trace(go.Scatter(x=chart_df_tech.index, y=chart_df_tech['MA_200'], mode='lines', name='MA 200', line=dict(color='red', width=1)));
                         fig_tech.update_layout(xaxis_rangeslider_visible=False, yaxis_title=f"Price ({currency_symbol})", legend_title="Legend"); st.plotly_chart(fig_tech, use_container_width=True)

                     st.divider(); st.subheader("Current Values"); col1_tech, col2_tech, col3_tech = st.columns(3); # ... (Display MA metrics) ...
                     with col1_tech: st.metric("20D MA", f"{currency_symbol}{df_tech['MA_20'].iloc[-1]:.2f}" if pd.notna(df_tech['MA_20'].iloc[-1]) else "N/A")
                     with col2_tech: st.metric("50D MA", f"{currency_symbol}{df_tech['MA_50'].iloc[-1]:.2f}" if pd.notna(df_tech['MA_50'].iloc[-1]) else "N/A")
                     with col3_tech: st.metric("200D MA", f"{currency_symbol}{df_tech['MA_200'].iloc[-1]:.2f}" if pd.notna(df_tech['MA_200'].iloc[-1]) else "N/A")
             else: st.info("Load data (Tab 1) first.")

        with tab5: # --- TAB 5: AI Summary ---
            st.subheader(f"🧩 AI Summary - {ticker}")
            # Check if Gemini is configured AND available
            if not GEMINI_AVAILABLE or gemini_model is None: st.warning("⚠️ Gemini not configured.")
            else:
                if st.button("Generate AI Summary", key="gen_summary", type="primary", use_container_width=True):
                    # Check required data exists for the CURRENT ticker
                    data_ready = ('prediction' in st.session_state and st.session_state.prediction is not None and
                                  'analyzed_df' in st.session_state and # Can be empty, must exist
                                  'stock_data' in st.session_state and st.session_state.stock_data is not None and
                                  st.session_state.get('loaded_ticker') == ticker and
                                  st.session_state.get('analysis_ticker') == ticker)
                    print(f"--- DEBUG (Tab 5) --- Data Ready Check: {data_ready}") # Debug check

                    if not data_ready: st.warning("⚠️ Run 'Full Analysis' first.")
                    else:
                        try: # ... (Generate Summary Logic - same as before) ...
                            with st.spinner("🧠 Gemini analyzing..."):
                                prediction_data = st.session_state.prediction; sentiment_df = st.session_state.analyzed_df; stock_df = st.session_state.stock_data
                                current_price = stock_df['Close'].iloc[-1]; day_change_pct = ((stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-2]) - 1) * 100 if len(stock_df)>1 and stock_df['Close'].iloc[-2] != 0 else 0; period_high = stock_df['High'].max()
                                lookback_desc = lookback_period # Use sidebar variable
                                trend = prediction_data.get('trend', 'N/A'); avg_sentiment = prediction_data.get('avg_sentiment', 0.0); target_30d = prediction_data.get('target_30d', 'N/A'); change_30d = prediction_data.get('predicted_change_30d', 'N/A'); confidence = prediction_data.get('confidence', 'N/A')
                                positive_count=0; neutral_count=0; negative_count=0
                                if sentiment_df is not None and not sentiment_df.empty and 'sentiment' in sentiment_df.columns: sentiment_counts = sentiment_df['sentiment'].value_counts(); positive_count = int(sentiment_counts.get('positive', 0) + sentiment_counts.get('POSITIVE', 0)); neutral_count = int(sentiment_counts.get('neutral', 0) + sentiment_counts.get('NEUTRAL', 0)); negative_count = int(sentiment_counts.get('negative', 0) + sentiment_counts.get('NEGATIVE', 0))
                                prompt = f"""
                                You are an expert financial analyst with access to the latest market data. Your task is to provide a comprehensive analysis of the {ticker} stock based on the following information:

                                - Current Price: {current_price}
                                - Day Change (%): {day_change_pct}
                                - Period High: {period_high}
                                - Lookback Period: {lookback_desc}
                                - Trend: {trend}
                                - Average Sentiment: {avg_sentiment}
                                - 30-Day Target: {target_30d}
                                - 30-Day Change (%): {change_30d}
                                - Confidence: {confidence}
                                - Positive Mentions: {positive_count}
                                - Neutral Mentions: {neutral_count}
                                - Negative Mentions: {negative_count}

                                Please provide your analysis in a clear and concise manner.
                                """
                                response = gemini_model.generate_content(prompt); st.markdown(response.text); st.session_state.gemini_summary = response.text; st.session_state.summary_ticker = ticker
                        except Exception as e: st.error(f"❌ Summary error: {e}"); st.exception(e)

                elif 'gemini_summary' in st.session_state and st.session_state.get('summary_ticker') == ticker: st.markdown(st.session_state.gemini_summary)
                else: st.info("Click 'Generate AI Summary'.")

        with tab6: # --- TAB 6: Company Profile ---
            st.subheader(f"🏢 Company Profile")
            # Display requires stock_info *for the current ticker*
            if 'stock_info' in st.session_state and st.session_state.loaded_ticker == ticker:
                 stock_info_display = st.session_state.stock_info
                 if stock_info_display:
                     # ... (Display profile info, financials, recommendations - same as before) ...
                     st.header(stock_info_display.get('longName', ticker))
                     col1_prof, col2_prof = st.columns(2);
                     with col1_prof: st.caption(f"**Sector:** {stock_info_display.get('sector', 'N/A')}"); st.caption(f"**Industry:** {stock_info_display.get('industry', 'N/A')}")
                     with col2_prof: st.caption(f"**Website:** {stock_info_display.get('website', 'N/A')}"); st.caption(f"**Country:** {stock_info_display.get('country', 'N/A')}")
                     st.divider(); st.subheader("Summary"); st.markdown(stock_info_display.get('longBusinessSummary', 'N/A'))
                     st.divider(); st.subheader("Financials"); col1_fin, col2_fin, col3_fin = st.columns(3);
                     mcap = stock_info_display.get('marketCap'); st.metric("Market Cap", f"{currency_symbol}{mcap/1e9:.2f}B" if mcap else "N/A")
                     eval = stock_info_display.get('enterpriseValue'); st.metric("Ent Value", f"{currency_symbol}{eval/1e9:.2f}B" if eval else "N/A")
                     tpe = stock_info_display.get('trailingPE'); st.metric("Trailing P/E", f"{tpe:.2f}" if isinstance(tpe, (int, float)) else 'N/A')
                     fpe = stock_info_display.get('forwardPE'); st.metric("Forward P/E", f"{fpe:.2f}" if isinstance(fpe, (int, float)) else 'N/A')
                     dyield = stock_info_display.get('dividendYield'); st.metric("Div Yield", f"{dyield * 100:.2f}%" if dyield else "N/A")
                     beta_val = stock_info_display.get('beta'); st.metric("Beta", f"{beta_val:.2f}" if isinstance(beta_val, (int, float)) else 'N/A')
                     st.divider(); st.subheader("Recommendations");
                     try:
                         with st.spinner("Loading..."): recs = yf.Ticker(ticker).recommendations
                         if recs is not None and not recs.empty: st.dataframe(recs.tail(10), use_container_width=True)
                         else: st.info("None available.")
                     except Exception as e: st.warning(f"Could not load: {e}")
                 else: st.warning(f"Profile info unavailable.")
            else: st.info("Load data (Tab 1) first.")
        # --- End Main Content Tabs ---

    # --- End Logged-in code ---

# --- Authentication States (Login Failed / Not Logged In but Routed Wrong) ---
# Redirect logic handled by the main routing check at the top
# # and logout logic within the main app state.
# elif st.session_state.get('authentication_status') == False: 
#     login_page() # Rerender login page with error
# elif st.session_state.get('authentication_status') is None and st.session_state.get('menu_selection') != 'Register': 
#     login_page()

# # --- Fallback for invalid state ---
# elif st.session_state.get('menu_selection') not in ['Login', 'Register', 'App']: # <-- Colon corrected here
#      st.error("Invalid app state. Resetting to login.")
#      print(f"--- ERROR --- Invalid menu_selection state detected: {st.session_state.get('menu_selection')}") # Debug print
#      st.session_state.menu_selection = 'Login' # Reset state variable used for routing
#      # Clear potentially conflicting auth state
#      keys_to_clear_fallback = ['authentication_status', 'name', 'username', 'watchlist']
#      for key in keys_to_clear_fallback:
#          if key in st.session_state: del st.session_state[key]
#      st.rerun()

# --- App Footer (Optional) ---
# st.markdown("---")
# st.caption("SynapseStreet vX.Y")
