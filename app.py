import streamlit as st
import pandas as pd
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_multiple_news  # updated function
from src.nlp_sentiment_predictor import nlp_sentiment_pipeline

st.set_page_config(page_title='SynapseStreet', layout='wide')
st.title('🧠 SynapseStreet')

# --- Predefined Tickers ---
POPULAR_TICKERS = [
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

st.sidebar.header('Controls')
run_scrape = st.sidebar.button('Fetch News & Predict')

st.header('Stock Data Explorer')

ticker = st.selectbox(
    'Search or Enter a Stock Ticker (e.g., AAPL, RELIANCE.NS)',
    options=[''] + POPULAR_TICKERS,
    index=0,
    placeholder="Type to search..."
).strip().upper()

# ----------------------------
# STOCK DATA FETCH SECTION
# ----------------------------
if ticker:
    st.markdown(f'**Selected Ticker:** `{ticker}`')

    if st.button(f'Load Historical Data for {ticker}', key='fetch_stock_data_btn'):
        try:
            with st.spinner(f'Fetching data for {ticker}...'):
                df = get_stock_data(ticker, start='2020-01-01')

            if df is None or df.empty:
                st.error(f'Could not find stock data for ticker: **{ticker}**.')
            else:
                st.success(f'Successfully fetched {len(df)} days of data.')

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.to_flat_index()
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                st.dataframe(df.head())

                if 'Close' in df.columns and not df['Close'].isnull().all():
                    st.subheader(f'Price History: {ticker}')
                    st.line_chart(df['Close'])
                else:
                    st.warning('Missing or invalid Close price data.')
        except Exception as e:
            st.error(f'Error fetching stock data: {e}')
else:
    st.info('Please select or enter a valid stock ticker.')

# ----------------------------
# REPLACED OLD NEWS SECTION
# ----------------------------
st.header('🧠 News Sentiment & Prediction')

if run_scrape:
    if not ticker:
        st.warning("Please select a stock first.")
    else:
        with st.spinner(f'Fetching latest news and predicting for {ticker}...'):
            try:
                # 🧱 Fetch fresh news and store to DB
                news_df = scrape_multiple_news(ticker, limit=10)

                if db_available and not news_df.empty:
                    from datetime import datetime
                    collection = db.get_collection("news_articles")
                    data_to_insert = news_df.to_dict(orient="records")
                    for item in data_to_insert:
                        item["ticker"] = ticker
                        item["timestamp"] = datetime.utcnow()
                    collection.insert_many(data_to_insert)

                # 🧠 Run NLP Sentiment + Prediction
                analyzed_df, prediction = nlp_sentiment_pipeline(ticker)

                if analyzed_df is None:
                    st.warning(prediction)
                else:
                    st.success(f"✅ Sentiment and prediction for {ticker} ready.")
                    st.subheader("📊 Predicted Stock Outlook")

                    st.markdown(f"""
                    **Ticker:** `{prediction['ticker']}`  
                    **Average Sentiment:** `{prediction['avg_sentiment']:.2f}`  
                    **Predicted Trend:** {prediction['trend']}  
                    _Generated on {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}_  
                    """)

                    st.divider()
                    st.write("Detailed Sentiment Breakdown:")
                    st.dataframe(analyzed_df[['title', 'sentiment', 'score']])
            except Exception as e:
                st.error(f"Error while running sentiment pipeline: {e}")
