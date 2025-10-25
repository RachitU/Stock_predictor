"""
SynapseStreet - AI-Powered Stock Analysis with Sentiment & ML Predictions
"""

import streamlit as st
import pandas as pd
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_multiple_news
from src.nlp_sentiment_predictor import nlp_sentiment_pipeline
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title='SynapseStreet - AI Stock Predictor',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🧠'
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2193b0, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #080808;
        color: #ffffff;
        border-radius: 5px;
        font-weight: 600;
        padding:2px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #2193b0, #6dd5ed);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown('<h1 class="main-header">🧠 SynapseStreet</h1>', unsafe_allow_html=True)
st.markdown("*AI-Powered Stock Analysis combining Sentiment & Machine Learning*")

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

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header('🎛️ Controls')
    
    ticker = st.selectbox(
        '📊 Select Stock Ticker',
        options=[''] + POPULAR_TICKERS,
        index=0,
        placeholder="Type to search...",
        help="Select or type a stock ticker symbol"
    ).strip().upper()
    
    st.divider()
    
    # Analysis Options
    st.subheader("Analysis Options")
    
    forecast_days = st.slider(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1,
        help="Number of days to forecast ahead"
    )
    
    lookback_period = st.selectbox(
        "Historical Data Period",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=3,
        help="Amount of historical data to fetch"
    )
    
    st.divider()
    
    # Action Buttons
    run_analysis = st.button(
        '🚀 Run Full Analysis',
        type='primary',
        use_container_width=True,
        help="Fetch news, analyze sentiment, and generate predictions"
    )
    
    refresh_data = st.button(
        '🔄 Refresh Data Only',
        use_container_width=True,
        help="Update stock data without running predictions"
    )
    
    st.divider()
    
    # Database Status
    if db_available:
        st.success("✅ Database Connected")
    else:
        st.warning("⚠️ Database Offline")
    
    # API Status
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API Ready")
    else:
        st.info("ℹ️ OpenAI API Not Configured")

# Main Content Area
if not ticker:
    # Landing Page
    st.info("👈 Please select a stock ticker from the sidebar to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📰 News Sentiment")
        st.write("Real-time sentiment analysis from multiple news sources using FinBERT")
    
    with col2:
        st.markdown("### 🤖 ML Predictions")
        st.write("XGBoost-powered forecasting with technical indicators and sentiment features")
    
    with col3:
        st.markdown("### 📈 Visualizations")
        st.write("Interactive charts showing historical data, predictions, and confidence bands")
    
    st.divider()
    
    # Popular Stocks Quick Access
    st.subheader("🔥 Quick Access - Popular Stocks")
    
    quick_cols = st.columns(6)
    quick_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
    
    for idx, qt in enumerate(quick_tickers):
        with quick_cols[idx]:
            if st.button(qt, use_container_width=True):
                st.session_state.selected_ticker = qt
                st.rerun()

else:
    # Main Analysis Interface
    st.markdown(f"## Analysis for **{ticker}**")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Historical Data",
        "🔮 AI Predictions",
        "📰 Data and Tally",
        "📈 Technical Indicators"
    ])
    
    # ----------------------------
    # TAB 1: Historical Data
    # ----------------------------
    with tab1:
        st.subheader(f"Historical Stock Data - {ticker}")
        
        if refresh_data or run_analysis or st.button("Load Historical Data", key="load_hist"):
            try:
                with st.spinner(f'Fetching historical data for {ticker}...'):
                    df = get_stock_data(ticker, start='2020-01-01')
                
                if df is None or df.empty:
                    st.error(f'Could not find stock data for ticker: **{ticker}**')
                else:
                    st.success(f'✅ Successfully fetched {len(df)} days of data')
                    
                    # Store in session state
                    st.session_state.stock_data = df
                    
                    # Handle MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.to_flat_index()
                        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'Close' in df.columns:
                            current_price = df['Close'].iloc[-1]
                            st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        if 'Close' in df.columns:
                            day_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                            day_change_pct = (day_change / df['Close'].iloc[-2]) * 100
                            st.metric("Day Change", f"${day_change:.2f}", f"{day_change_pct:+.2f}%")
                    
                    with col3:
                        if 'High' in df.columns and 'Low' in df.columns:
                            high_52w = df['High'].tail(252).max()
                            st.metric("52W High", f"${high_52w:.2f}")
                    
                    with col4:
                        if 'Volume' in df.columns:
                            avg_volume = df['Volume'].tail(20).mean()
                            st.metric("Avg Volume (20D)", f"{avg_volume/1e6:.2f}M")
                    
                    # Data table
                    st.divider()
                    st.dataframe(
                        df.tail(100),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Price chart
                    if 'Close' in df.columns and not df['Close'].isnull().all():
                        st.divider()
                        st.subheader("Price History")
                        st.line_chart(df['Close'].tail(252), use_container_width=True)
                    
            except Exception as e:
                st.error(f'❌ Error fetching stock data: {e}')
    
    # ----------------------------
    # TAB 2: AI Predictions
    # ----------------------------
    with tab2:
        st.subheader(f"🔮 AI-Powered Price Predictions - {ticker}")
        
        if run_analysis or st.button("Generate Predictions", key="gen_pred", type="primary"):
            try:
                with st.spinner(f'🧠 Running AI analysis for {ticker}...'):
                    # Run the full NLP sentiment pipeline with ML predictions
                    analyzed_df, prediction, fig = nlp_sentiment_pipeline(ticker, forecast_days=forecast_days)
                
                if fig is None:
                    st.error("Could not generate predictions. Please check the ticker symbol and try again.")
                else:
                    # Store in session state
                    st.session_state.prediction = prediction
                    st.session_state.prediction_fig = fig
                    st.session_state.analyzed_df = analyzed_df
                    
                    # Display prediction metrics
                    st.success("✅ Prediction generated successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        trend_emoji = "📈" if "Bullish" in prediction.get('trend', '') else "📉" if "Bearish" in prediction.get('trend', '') else "➖"
                        st.metric("Trend", prediction.get('trend', 'N/A'))
                    
                    with col2:
                        sentiment = prediction.get('avg_sentiment', 0)
                        st.metric("Sentiment Score", f"{sentiment:+.3f}")
                    
                    with col3:
                        st.metric(
                            "7-Day Target",
                            prediction.get('target_7d', 'N/A'),
                            prediction.get('predicted_change_7d', 'N/A')
                        )
                    
                    with col4:
                        st.metric(
                            "30-Day Target",
                            prediction.get('target_30d', 'N/A'),
                            prediction.get('predicted_change_30d', 'N/A')
                        )
                    
                    # Display confidence
                    st.divider()
                    confidence = prediction.get('confidence', 'Unknown')
                    if confidence == 'High':
                        st.success(f"🎯 Confidence Level: **{confidence}**")
                    elif confidence == 'Medium':
                        st.info(f"📊 Confidence Level: **{confidence}**")
                    else:
                        st.warning(f"⚠️ Confidence Level: **{confidence}**")
                    
                    # Display the prediction visualization
                    st.divider()
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional insights
                    with st.expander("📋 Prediction Details"):
                        st.json(prediction)
                    
                    # Download option
                    st.divider()
                    if st.button("💾 Save Prediction Chart"):
                        import os
                        os.makedirs("predictions", exist_ok=True)
                        fig.write_html(f"predictions/{ticker}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                        st.success(f"✅ Chart saved to predictions/{ticker}_forecast.html")
                        
            except Exception as e:
                st.error(f"❌ Error generating predictions: {e}")
                st.exception(e)
        
        # Display cached prediction if available
        elif 'prediction_fig' in st.session_state:
            st.info("Showing cached prediction. Click 'Generate Predictions' to update.")
            st.plotly_chart(st.session_state.prediction_fig, use_container_width=True)
    
    # ----------------------------
    # TAB 3: Sentiment Analysis
    # ----------------------------
    with tab3:
        st.subheader(f"📰 News Sentiment Analysis - {ticker}")
        
        if run_analysis or st.button("Fetch Latest News", key="fetch_news"):
            try:
                with st.spinner(f'Fetching and analyzing news for {ticker}...'):
                    # Fetch fresh news
                    news_df = scrape_multiple_news(ticker, limit=10)
                    
                    if news_df.empty:
                        st.warning("No recent news found for this ticker.")
                    else:
                        st.success(f"✅ Found {len(news_df)} news articles")
                        
                        # Store to database if available
                        if db_available:
                            collection = db.get_collection("news_articles")
                            data_to_insert = news_df.to_dict(orient="records")
                            for item in data_to_insert:
                                item["ticker"] = ticker
                                item["timestamp"] = datetime.utcnow()
                            collection.insert_many(data_to_insert)
                            st.info("📦 News articles saved to database")
                        
                        # Display analyzed sentiment from prediction
                        if 'analyzed_df' in st.session_state and not st.session_state.analyzed_df.empty:
                            analyzed_df = st.session_state.analyzed_df
                            
                            # Sentiment distribution
                            st.divider()
                            st.subheader("Sentiment Distribution")
                            
                            sentiment_counts = analyzed_df['sentiment'].value_counts()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                positive = sentiment_counts.get('positive', 0) + sentiment_counts.get('POSITIVE', 0)
                                st.metric("🟢 Positive", positive)
                            with col2:
                                neutral = sentiment_counts.get('neutral', 0) + sentiment_counts.get('NEUTRAL', 0)
                                st.metric("🟡 Neutral", neutral)
                            with col3:
                                negative = sentiment_counts.get('negative', 0) + sentiment_counts.get('NEGATIVE', 0)
                                st.metric("🔴 Negative", negative)
                            
                            # Detailed breakdown
                            st.divider()
                            st.subheader("Detailed Sentiment Breakdown")
                            
                            display_df = analyzed_df[['title', 'sentiment', 'score', 'sentiment_value', 'source']].copy()
                            
                            # Color code sentiment
                            def highlight_sentiment(row):
                                if row['sentiment_value'] > 0:
                                    return ['background-color: #008B8B'] * len(row)
                                elif row['sentiment_value'] < 0:
                                    return ['background-color: #E44D2E'] * len(row)
                                else:
                                    return ['background-color: #000000'] * len(row)
                            
                            st.dataframe(
                                display_df.style.apply(highlight_sentiment, axis=1),
                                use_container_width=True,
                                height=400
                            )
                        else:
                            st.info("Run 'Generate Predictions' in the AI Predictions tab to see sentiment analysis.")
                            
            except Exception as e:
                st.error(f"❌ Error analyzing sentiment: {e}")
        
        # Display cached sentiment if available
        elif 'analyzed_df' in st.session_state and not st.session_state.analyzed_df.empty:
            st.info("Showing cached sentiment analysis. Click 'Fetch Latest News' to update.")
            analyzed_df = st.session_state.analyzed_df
            st.dataframe(
                analyzed_df[['title', 'sentiment', 'score', 'sentiment_value']],
                use_container_width=True
            )
    
    # ----------------------------
    # TAB 4: Technical Indicators
    # ----------------------------
    with tab4:
        st.subheader(f"📈 Technical Indicators - {ticker}")
        
        if 'stock_data' in st.session_state:
            df = st.session_state.stock_data
            
            # Calculate basic indicators
            if 'Close' in df.columns:
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                df['MA_200'] = df['Close'].rolling(window=200).mean()
                
                # Display chart
                st.line_chart(df[['Close', 'MA_20', 'MA_50', 'MA_200']].tail(252))
                
                # Current indicator values
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("20-Day MA", f"${df['MA_20'].iloc[-1]:.2f}")
                with col2:
                    st.metric("50-Day MA", f"${df['MA_50'].iloc[-1]:.2f}")
                with col3:
                    st.metric("200-Day MA", f"${df['MA_200'].iloc[-1]:.2f}")
            else:
                st.warning("Load historical data first to view technical indicators.")
        else:
            st.info("Load historical data in the 'Historical Data' tab to view technical indicators.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>SynapseStreet</strong> · AI-Powered Stock Intelligence Platform</p>
    <p style='font-size: 0.9rem;'>Powered by FinBERT · XGBoost · NetworkX · Plotly · Streamlit</p>
    <p style='font-size: 0.8rem;'>⚠️ Disclaimer: This tool is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
