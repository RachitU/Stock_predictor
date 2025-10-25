# app.py - Streamlit launcher at project root
import streamlit as st
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_twitter, scrape_news
from src.nlp_analysis import analyze_sentiment, process_document
from src.insights import summarize_insight

st.set_page_config(page_title='SynapseStreet', layout='wide')
st.title('🧠 SynapseStreet — Working Demo')

st.sidebar.header('Controls')
ticker = st.sidebar.text_input('Ticker', 'AAPL')
run_scrape = st.sidebar.button('Scrape Sample Data')

st.header('Stock Data (yfinance)')
if st.button('Fetch stock data for ' + ticker):
    df = get_stock_data(ticker, start='2020-01-01')
    if df is None or df.empty:
        st.write('No data returned.')
    else:
        st.line_chart(df['Close'])

st.header('Quick Scrape & NLP (sample)')
if run_scrape:
    st.info('Running sample scrapes (Twitter + News) — results stored to DB if configured.')
    try:
        t = scrape_twitter('Tesla', limit=10)
        st.write('Scraped tweets:', t.head(5).to_dict(orient='records'))
    except Exception as e:
        st.error('Twitter scrape failed: ' + str(e))
    try:
        news_text = scrape_news('https://www.reuters.com/technology/')
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

st.write('---')
st.write('This is a minimal working package. Check README for setup steps.')
