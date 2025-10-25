import streamlit as st
import pandas as pd
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_multiple_news
from src.nlp_sentiment_predictor import nlp_sentiment_pipeline
from datetime import datetime

st.set_page_config(page_title='SynapseStreet', layout='wide')
st.title('🧠 SynapseStreet')

# --- Full Predefined Tickers (US + Indian) ---
POPULAR_TICKERS = [
    # US Stocks (FAANGM, Tech, Internet)
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 
    'TSLA', 'AMD', 'INTC', 'CSCO', 'QCOM',
    # US Finance, Banking, Payments
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP',
    # US Healthcare & Pharma
    'JNJ', 'PFE', 'LLY', 'UNH', 'MRK', 'ABBV', 'TMO',
    # US Consumer Staples, Retail & Industrials
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'LOW', 'NKE', 'MCD',
    'BA', 'CAT', 'GE', 'LMT',
    # US Energy & Utilities
    'XOM', 'CVX', 'DUK', 'NEE', 

    # Indian Stocks (NSE, use .NS suffix)
    # IT Services
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
    # Banking & Finance
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS',
    # Auto, Manufacturing, Infrastructure
    'RELIANCE.NS', 'LT.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'ULTRACEMCO.NS',
    # Pharma & Healthcare
    'SUNPHARMA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS',
    # FMCG & Others
    'HUL.NS', 'ITC.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'BHARTIARTL.NS', 'TITAN.NS', 'ADANIENT.NS'
]

st.sidebar.header('Controls')
run_scrape = st.sidebar.button('Fetch News & Predict')

# ----------------------------
# Stock selection
# ----------------------------
ticker = st.selectbox(
    'Search or Enter a Stock Ticker (e.g., AAPL, RELIANCE.NS)',
    options=[''] + POPULAR_TICKERS,
    index=0,
    placeholder="Type to search..."
).strip().upper()

# ----------------------------
# STOCK DATA SECTION
# ----------------------------
if ticker:
    st.markdown(f'**Selected Ticker:** `{ticker}`')
    if st.button(f'Load Historical Data for {ticker}', key='fetch_stock_data_btn'):
        try:
            with st.spinner(f'Fetching data for {ticker}...'):
                df = get_stock_data(ticker, start='2020-01-01')
            if df is None or df.empty:
                st.error(f'No stock data for {ticker}.')
            else:
                st.success(f'Fetched {len(df)} rows.')
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.to_flat_index()
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                st.dataframe(df.head())
                if 'Close' in df.columns and not df['Close'].isnull().all():
                    st.line_chart(df['Close'])
        except Exception as e:
            st.error(f'Error fetching stock data: {e}')
else:
    st.info('Please select a valid stock ticker.')

# ----------------------------
# NEWS + SENTIMENT PIPELINE
# ----------------------------
st.header('🧠 News Sentiment & Prediction')

if run_scrape:
    if not ticker:
        st.warning("Please select a stock first.")
    else:
        with st.spinner(f'Fetching news and predicting for {ticker}...'):
            try:
                # Fetch news (Google News + Zerodha Pulse for Indian stocks)
                news_df = scrape_multiple_news(ticker, limit=10, max_age_days=5)

                # Store in MongoDB
                if db_available and not news_df.empty:
                    collection = db.get_collection("news_articles")
                    data_to_insert = news_df.to_dict(orient="records")
                    for item in data_to_insert:
                        item["ticker"] = ticker
                        item["timestamp"] = item.get("timestamp", datetime.utcnow())
                    collection.insert_many(data_to_insert)

                # NLP Sentiment + Prediction
                analyzed_df, prediction = nlp_sentiment_pipeline(ticker)

                if analyzed_df is None:
                    st.warning(prediction)
                else:
                    st.success(f"✅ Sentiment & prediction ready for {ticker}.")
                    st.subheader("📊 Predicted Stock Outlook")
                    st.markdown(f"""
                    **Ticker:** `{prediction['ticker']}`  
                    **Average Sentiment:** `{prediction['avg_sentiment']:.2f}`  
                    **Predicted Trend:** {prediction['trend']}  
                    _Generated on {prediction['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}_  
                    """)

                    st.divider()
                    st.write("Detailed Sentiment Breakdown (All Sources):")
                    st.dataframe(analyzed_df[['source','title','sentiment','score']])

                    # Tally Positive / Neutral / Negative
                    tally = analyzed_df['sentiment'].value_counts().reindex(['positive','neutral','negative'], fill_value=0)
                    st.subheader("📊 Sentiment Tally")
                    st.markdown(f"""
                    **Positive:** {tally['positive']}  
                    **Neutral:** {tally['neutral']}  
                    **Negative:** {tally['negative']}
                    """)

            except Exception as e:
                st.error(f"Error in sentiment pipeline: {e}")
