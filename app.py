# app.py - Streamlit launcher at project root
import streamlit as st
import pandas as pd # Import pandas for data manipulation
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_twitter, scrape_news
from src.nlp_analysis import analyze_sentiment, process_document
#from src.insights import summarize_insight

st.set_page_config(page_title='SynapseStreet', layout='wide')
st.title('🧠 SynapseStreet')

# --- Re-add Predefined Tickers for Autocomplete/Search Suggestions (MASSIVELY EXPANDED) ---
POPULAR_TICKERS = [
    # US Stocks (FAANGM, Tech, Internet)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 
    'TSLA', 'AMD', 'INTC', 'CSCO', 'QCOM',

    # US Finance, Banking, and Payments
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP',

    # US Healthcare & Pharma
    'JNJ', 'PFE', 'LLY', 'UNH', 'MRK', 'ABBV', 'TMO',

    # US Consumer Staples, Retail & Industrials
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'LOW', 'NKE', 'MCD',
    'BA', 'CAT', 'GE', 'LMT',

    # US Energy & Utilities
    'XOM', 'CVX', 'DUK', 'NEE', 

    # Indian Stocks (NSE, use .NS suffix for yfinance)
    # IT Services
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',

    # Banking & Finance
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS',

    # Auto, Manufacturing, and Infrastructure
    'RELIANCE.NS', 'LT.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'ULTRACEMCO.NS',

    # Pharma & Healthcare
    'SUNPHARMA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS',

    # FMCG & Others
    'HUL.NS', 'ITC.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'BHARTIARTL.NS', 'TITAN.NS', 'ADANIENT.NS'
]

# --- Input Control ---
st.sidebar.header('Controls')

# The button for scrape is kept in the sidebar
run_scrape = st.sidebar.button('Scrape Sample Data')


st.header('Stock Data Explorer')

# New: Add the popular tickers list inside an expander for reference
with st.expander("View Popular Tickers/Keywords"):
    st.code(", ".join(POPULAR_TICKERS), language='text')

# --- MERGED INPUT BOX (using text_input for guaranteed custom input) ---
# This single text input box now handles both the stock ticker and the scrape keyword.
# It is replaced by st.text_input to allow any custom input without the "no result" error.
ticker_or_keyword_raw = st.text_input(
    'Enter Ticker / Scrape Keyword',
    # Use a default ticker from the list
    value='AAPL', 
    max_chars=30
).strip()

# Use the selected/typed value as the main ticker/keyword
ticker = ticker_or_keyword_raw.upper()

# The main area now only uses the 'ticker' variable
if ticker:
    st.markdown(f'**Current Ticker/Keyword:** `{ticker}`')
    
    # Button is dynamic based on the selected ticker
    if st.button(f'Load Historical Data for {ticker}', key='fetch_stock_data_btn'):
        try:
            with st.spinner(f'Fetching data for {ticker}...'):
                # Fetch data for the last few years from 2020-01-01
                df = get_stock_data(ticker, start='2020-01-01')
            
            if df is None or df.empty:
                st.error(f'Could not find stock data for ticker: **{ticker}**. Please check the ticker symbol (e.g., use `.NS` for Indian stocks like `RELIANCE.NS`, or `.BO` for BSE stocks).')
            else:
                # --- CHANGE IS HERE: Use df.tail() to show latest data ---
                st.success(f'Successfully fetched {len(df)} days of data. Showing latest 5 rows below:')
                
                # FIX: Flatten the column index if it is a MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.to_flat_index()
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Show the LATEST data first for inspection
                st.dataframe(df.tail())

                # Check for valid data before plotting
                if 'Close' not in df.columns:
                     st.error('The fetched data is missing the required "Close" price column after processing.')
                elif df['Close'].isnull().all():
                    st.warning('Data was fetched, but the "Close" column is entirely empty (NaN). The chart will be blank.')
                elif (df['Close'] == 0).all():
                    st.warning('Data was fetched, but the "Close" column is entirely zero. The chart will be a flat line at 0.')
                else:
                    # Data looks valid, proceed with charting
                    st.subheader(f'Price History: {ticker} (Closing Price)')
                    st.line_chart(df['Close'])
                    
        except Exception as e:
            st.error(f'An error occurred while fetching stock data: {e}')
else:
    st.info('Please enter a valid stock ticker or keyword to proceed.')

# --- Quick Scrape & NLP Section (Updated to use the single input) ---
st.header('Quick Scrape & NLP (sample)')
if run_scrape:
    if not ticker:
        st.warning('Please enter a keyword in the input box before running the scrape.')
    else:
        st.info(f'Running sample scrapes (Twitter + News) for keyword: **{ticker}** — results stored to DB if configured.')
        try:
            # Uses the unified 'ticker' for scraping
            t = scrape_twitter(ticker, limit=10)
            st.write('Scraped tweets:', t.head(5).to_dict(orient='records'))
        except Exception as e:
            st.error('Twitter scrape failed: ' + str(e))
        try:
            news_text = scrape_news('https://m.economictimes.com/markets')
            st.write('Scraped news snippet:', news_text[:400])
        except Exception as e:
            st.error('News scrape failed: ' + str(e))

        # process a few tweets (if any)
        if db_available:
            samples = list(db.get_collection('tweets').find().limit(5))
            for s in samples:
                r = process_document(s)
                st.write(r)
        else:
            st.write('DB not configured; skipping NLP pipeline storage.')
