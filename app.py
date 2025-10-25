# app.py - Streamlit launcher at project root
import streamlit as st
import pandas as pd # Import pandas for data manipulation
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_twitter, scrape_news
from src.nlp_analysis import analyze_sentiment, process_document
from src.insights import summarize_insight

st.set_page_config(page_title='SynapseStreet', layout='wide')
st.title('🧠 SynapseStreet')

# --- Predefined Tickers for Autocomplete/Search Suggestions (MASSIVELY EXPANDED) ---
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

st.sidebar.header('Controls')
run_scrape = st.sidebar.button('Scrape Sample Data')


st.header('Stock Data Explorer')

# Use st.selectbox for search/autocomplete functionality.
selected_ticker = st.selectbox(
    'Search or Select a Stock Ticker (e.g., AAPL, RELIANCE.NS)',
    options=[''] + POPULAR_TICKERS,
    index=0
)

# Allow manual override if the user types something not in the list
custom_ticker = st.text_input('Or enter a Custom Ticker', value='')

# Determine the final ticker to use
ticker = custom_ticker.upper().strip() if custom_ticker.strip() else selected_ticker.upper().strip()

# Check if a ticker is available to fetch
if ticker:
    st.markdown(f'**Selected Ticker:** `{ticker}`')
    
    # Button is now dynamic based on the selected ticker
    if st.button(f'Load Historical Data for {ticker}', key='fetch_stock_data_btn'):
        try:
            with st.spinner(f'Fetching data for {ticker}...'):
                # Fetch data for the last few years from 2020-01-01
                df = get_stock_data(ticker, start='2020-01-01')
            
            if df is None or df.empty:
                st.error(f'Could not find stock data for ticker: **{ticker}**. Please check the ticker symbol (e.g., use `.NS` for Indian stocks like `RELIANCE.NS`, or `.BO` for BSE stocks).')
            else:
                st.success(f'Successfully fetched {len(df)} days of data. Showing first 5 rows below:')
                
                # FIX: Flatten the column index. The yfinance call sometimes returns a MultiIndex
                # even for a single ticker, which causes issues with df['Close'] access.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.to_flat_index()
                    # Rename the columns if they became tuples, like ('Close', 'TATAMOTORS.NS')
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Show the raw data first for inspection
                st.dataframe(df.head())

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
    st.info('Please select or enter a valid stock ticker to load its historical data.')

# --- Original Quick Scrape & NLP Section (Unchanged) ---
st.header('Quick Scrape & NLP (sample)')
if run_scrape:
    st.info('Running sample scrapes (Twitter + News) — results stored to DB if configured.')
    try:
        t = scrape_twitter('Tesla', limit=10)
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

st.header('AI Insight (OpenAI)')
user_text = st.text_area('Paste headline / tweet / note to summarize', 'Tesla hiring surge reported.')
if st.button('Generate Insight'):
    result = summarize_insight(user_text)
    st.write(result)
