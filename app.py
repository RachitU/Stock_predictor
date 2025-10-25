"""
SynapseStreet - AI-Powered Stock Analysis with Sentiment & ML Predictions
Complete Integration with Enhanced UI, Tabs, and Sentiment Analysis
"""

import streamlit as st
import pandas as pd
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_multiple_news
from src.nlp_sentiment_predictor import nlp_sentiment_pipeline
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title='SynapseStreet - AI Stock Predictor',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🧠'
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2193b0, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
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
        background-color: #f0f2f6;
        border-radius: 5px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #2193b0, #6dd5ed);
        color: white;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE & HEADER
# ============================================================================

st.markdown('<h1 class="main-header">🧠 SynapseStreet</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Stock Analysis combining News Sentiment & Machine Learning</p>', unsafe_allow_html=True)

# ============================================================================
# PREDEFINED TICKERS
# ============================================================================

POPULAR_TICKERS = [
    # US Stocks - Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 
    'TSLA', 'AMD', 'INTC', 'CSCO', 'QCOM',
    # US - Finance & Banking
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP',
    # US - Healthcare
    'JNJ', 'PFE', 'LLY', 'UNH', 'MRK', 'ABBV', 'TMO',
    # US - Consumer & Retail
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'HD', 'LOW', 'NKE', 'MCD',
    # US - Industrial & Energy
    'BA', 'CAT', 'GE', 'LMT', 'XOM', 'CVX', 'DUK', 'NEE',
    # Indian Stocks - IT
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
    # Indian - Banking
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS',
    # Indian - Manufacturing & Infrastructure
    'RELIANCE.NS', 'LT.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'ULTRACEMCO.NS',
    # Indian - Pharma
    'SUNPHARMA.NS', 'DRREDDY.NS', 'APOLLOHOSP.NS',
    # Indian - FMCG & Others
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
>>>>>>> 6cdf3a80c99f767c3213d33228a58c56aa9b1129
    'HUL.NS', 'ITC.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'BHARTIARTL.NS', 'TITAN.NS', 'ADANIENT.NS'
]

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.header('🎛️ Control Panel')
    
    # Stock Selection
    ticker = st.selectbox(
        '📊 Select Stock Ticker',
        options=[''] + POPULAR_TICKERS,
        index=0,
        placeholder="Type to search...",
        help="Select or type a stock ticker symbol (e.g., AAPL, RELIANCE.NS)"
    ).strip().upper()
    
    st.divider()
    
    # Analysis Options
    st.subheader("⚙️ Analysis Settings")
    
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
    
    news_limit = st.slider(
        "Max News Articles",
        min_value=5,
        max_value=50,
        value=10,
        help="Maximum number of news articles to fetch"
    )
    
    news_age_days = st.slider(
        "News Age (days)",
        min_value=1,
        max_value=30,
        value=5,
        help="Fetch news from last N days"
    )
    
    st.divider()
    
    # Action Buttons
    run_full_analysis = st.button(
        '🚀 Run Full Analysis',
        type='primary',
        use_container_width=True,
        help="Fetch data, analyze sentiment, generate predictions, and create visualizations"
    )
    
    fetch_news_only = st.button(
        '📰 Fetch News Only',
        use_container_width=True,
        help="Only fetch and analyze news sentiment"
    )
    
    refresh_data = st.button(
        '🔄 Refresh Stock Data',
        use_container_width=True,
        help="Update stock data without running predictions"
    )
    
    st.divider()
    
    # System Status
    st.subheader("🔌 System Status")
    if db_available:
        st.success("✅ Database Connected")
    else:
        st.warning("⚠️ Database Offline (CSV mode)")
    
    if OPENAI_API_KEY:
        st.success("✅ OpenAI API Ready")
    else:
        st.info("ℹ️ OpenAI API Not Configured")
    
    st.divider()
    
    # Info
    st.caption("💡 **Quick Tips:**")
    st.caption("• Run Full Analysis for complete insights")
    st.caption("• Switch between tabs to explore different views")
    st.caption("• US stocks: use symbol only (AAPL)")
    st.caption("• Indian stocks: add .NS (RELIANCE.NS)")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if not ticker:
    # ========================================================================
    # LANDING PAGE
    # ========================================================================
    
    st.info("👈 Please select a stock ticker from the sidebar to begin analysis")
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📰 Multi-Source News")
        st.write("Aggregates news from Google News, Yahoo Finance, and specialized sources")
        st.write("🔹 FinBERT sentiment analysis")
        st.write("🔹 Real-time updates")
    
    with col2:
        st.markdown("### 🤖 ML Predictions")
        st.write("XGBoost-powered forecasting with advanced features")
        st.write("🔹 Technical indicators (RSI, MACD, BB)")
        st.write("🔹 Sentiment integration")
    
    with col3:
        st.markdown("### 📈 Interactive Charts")
        st.write("Beautiful visualizations with Plotly")
        st.write("🔹 Prediction confidence bands")
        st.write("🔹 Sentiment timeline")
    
    st.divider()
    
    # Quick Access Buttons
    st.subheader("🔥 Quick Access - Popular Stocks")
    
    # US Stocks
    st.markdown("**🇺🇸 US Markets**")
    us_cols = st.columns(6)
    us_quick = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN']
    for idx, qt in enumerate(us_quick):
        with us_cols[idx]:
            if st.button(qt, use_container_width=True, key=f"us_{qt}"):
                st.session_state.selected_ticker = qt
                st.rerun()
    
    # Indian Stocks
    st.markdown("**🇮🇳 Indian Markets**")
    in_cols = st.columns(6)
    in_quick = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS']
    for idx, qt in enumerate(in_quick):
        with in_cols[idx]:
            if st.button(qt.replace('.NS', ''), use_container_width=True, key=f"in_{qt}"):
                st.session_state.selected_ticker = qt
                st.rerun()

else:
    # ========================================================================
    # MAIN ANALYSIS INTERFACE
    # ========================================================================
    
    st.markdown(f"## 📊 Analysis Dashboard: **{ticker}**")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Historical Data",
        "🔮 AI Predictions",
        "📰 Sentiment Analysis",
        "📈 Technical Indicators",
        "📋 Summary Report"
    ])
    
    # ========================================================================
    # TAB 1: HISTORICAL DATA
    # ========================================================================
    
    with tab1:
        st.subheader(f"📊 Historical Stock Data - {ticker}")
        
        if refresh_data or run_full_analysis or st.button("📥 Load Historical Data", key="load_hist"):
            try:
                with st.spinner(f'⏳ Fetching historical data for {ticker}...'):
                    df = get_stock_data(ticker, start='2020-01-01')
                
                if df is None or df.empty:
                    st.error(f'❌ Could not find stock data for ticker: **{ticker}**')
                    st.info("💡 Make sure you're using the correct ticker format:")
                    st.write("• US stocks: `AAPL`, `MSFT`, `GOOGL`")
                    st.write("• Indian stocks: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`")
                else:
                    st.success(f'✅ Successfully fetched **{len(df)}** days of data')
                    
                    # Store in session state
                    st.session_state.stock_data = df
                    
                    # Handle MultiIndex columns
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.to_flat_index()
                        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                    
                    # ========== METRICS ROW ==========
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        if 'Close' in df.columns:
                            current_price = df['Close'].iloc[-1]
                            st.metric("💰 Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        if 'Close' in df.columns and len(df) > 1:
                            day_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                            day_change_pct = (day_change / df['Close'].iloc[-2]) * 100
                            st.metric("📈 Day Change", f"${day_change:.2f}", f"{day_change_pct:+.2f}%")
                    
                    with col3:
                        if 'High' in df.columns:
                            high_52w = df['High'].tail(252).max()
                            st.metric("🔝 52W High", f"${high_52w:.2f}")
                    
                    with col4:
                        if 'Low' in df.columns:
                            low_52w = df['Low'].tail(252).min()
                            st.metric("📉 52W Low", f"${low_52w:.2f}")
                    
                    with col5:
                        if 'Volume' in df.columns:
                            avg_volume = df['Volume'].tail(20).mean()
                            st.metric("📊 Avg Volume (20D)", f"{avg_volume/1e6:.2f}M")
                    
                    # ========== PRICE CHART ==========
                    st.divider()
                    
                    if 'Close' in df.columns and not df['Close'].isnull().all():
                        st.subheader("📈 Price History")
                        
                        # Create interactive chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['Date'] if 'Date' in df.columns else df.index,
                            y=df['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='#2193b0', width=2)
                        ))
                        
                        fig.update_layout(
                            height=500,
                            hovermode='x unified',
                            showlegend=True,
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # ========== DATA TABLE ==========
                    st.divider()
                    st.subheader("📋 Recent Data (Last 100 Days)")
                    
                    display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] if 'Date' in df.columns else ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_cols = [col for col in display_cols if col in df.columns]
                    
                    st.dataframe(
                        df[available_cols].tail(100),
                        use_container_width=True,
                        height=400
                    )
                    
            except Exception as e:
                st.error(f'❌ Error fetching stock data: {e}')
                st.exception(e)
    
    # ========================================================================
    # TAB 2: AI PREDICTIONS
    # ========================================================================
    
    with tab2:
        st.subheader(f"🔮 AI-Powered Price Predictions - {ticker}")
        
        if run_full_analysis or st.button("🎯 Generate Predictions", key="gen_pred", type="primary"):
            try:
                with st.spinner(f'🧠 Running AI analysis for {ticker}... This may take a minute...'):
                    # Run the full NLP sentiment pipeline with ML predictions
                    analyzed_df, prediction, fig = nlp_sentiment_pipeline(ticker, forecast_days=forecast_days)
                
                if fig is None:
                    st.error("❌ Could not generate predictions. Please check the ticker symbol and try again.")
                    if isinstance(prediction, dict) and 'error' in prediction:
                        st.error(f"Error details: {prediction['error']}")
                else:
                    # Store in session state
                    st.session_state.prediction = prediction
                    st.session_state.prediction_fig = fig
                    st.session_state.analyzed_df = analyzed_df
                    
                    # ========== SUCCESS & METRICS ==========
                    st.success("✅ Prediction generated successfully!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        trend = prediction.get('trend', 'N/A')
                        st.metric("📊 Predicted Trend", trend)
                    
                    with col2:
                        sentiment = prediction.get('avg_sentiment', 0)
                        sentiment_emoji = "😊" if sentiment > 0.2 else "😐" if sentiment > -0.2 else "😟"
                        st.metric(f"{sentiment_emoji} Sentiment", f"{sentiment:+.3f}")
                    
                    with col3:
                        st.metric(
                            "🎯 7-Day Target",
                            prediction.get('target_7d', 'N/A'),
                            prediction.get('predicted_change_7d', 'N/A')
                        )
                    
                    with col4:
                        st.metric(
                            "🚀 30-Day Target",
                            prediction.get('target_30d', 'N/A'),
                            prediction.get('predicted_change_30d', 'N/A')
                        )
                    
                    # ========== CONFIDENCE INDICATOR ==========
                    st.divider()
                    
                    confidence = prediction.get('confidence', 'Unknown')
                    if confidence == 'High':
                        st.success(f"🎯 **Confidence Level:** {confidence} - Strong signal alignment detected")
                    elif confidence == 'Medium':
                        st.info(f"📊 **Confidence Level:** {confidence} - Moderate signal strength")
                    else:
                        st.warning(f"⚠️ **Confidence Level:** {confidence} - Weak or conflicting signals")
                    
                    # ========== PREDICTION CHART ==========
                    st.divider()
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ========== PREDICTION DETAILS ==========
                    with st.expander("📋 View Detailed Prediction Data"):
                        st.json(prediction)
                    
                    # ========== SAVE OPTION ==========
                    st.divider()
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("💾 Save Chart as HTML"):
                            import os
                            os.makedirs("predictions", exist_ok=True)
                            filename = f"predictions/{ticker}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                            fig.write_html(filename)
                            st.success(f"✅ Chart saved to `{filename}`")
                    
            except Exception as e:
                st.error(f"❌ Error generating predictions: {e}")
                st.exception(e)
        
        # Display cached prediction if available
        elif 'prediction_fig' in st.session_state:
            st.info("ℹ️ Showing cached prediction. Click 'Generate Predictions' to update with latest data.")
            st.plotly_chart(st.session_state.prediction_fig, use_container_width=True)
            
            if 'prediction' in st.session_state:
                pred = st.session_state.prediction
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Trend", pred.get('trend', 'N/A'))
                with col2:
                    st.metric("Sentiment", f"{pred.get('avg_sentiment', 0):+.3f}")
                with col3:
                    st.metric("7-Day Target", pred.get('target_7d', 'N/A'))
                with col4:
                    st.metric("30-Day Target", pred.get('target_30d', 'N/A'))
    
    # ========================================================================
    # TAB 3: SENTIMENT ANALYSIS
    # ========================================================================
    
    with tab3:
        st.subheader(f"📰 News Sentiment Analysis - {ticker}")
        
        if run_full_analysis or fetch_news_only or st.button("📡 Fetch Latest News", key="fetch_news"):
            try:
                with st.spinner(f'⏳ Fetching and analyzing news for {ticker}...'):
                    # Fetch fresh news
                    news_df = scrape_multiple_news(ticker, limit=news_limit, max_age_days=news_age_days)
                    
                    if news_df.empty:
                        st.warning(f"⚠️ No recent news found for **{ticker}** in the last {news_age_days} days.")
                        st.info("💡 Try:")
                        st.write("• Increasing the 'News Age' slider in the sidebar")
                        st.write("• Checking if the ticker symbol is correct")
                        st.write("• Selecting a more popular stock")
                    else:
                        st.success(f"✅ Found **{len(news_df)}** news articles from multiple sources")
                        
                        # Store to session state
                        st.session_state.news_df = news_df
                        
                        # Store to database if available
                        if db_available:
                            try:
                                collection = db.get_collection("news_articles")
                                data_to_insert = news_df.to_dict(orient="records")
                                for item in data_to_insert:
                                    item["ticker"] = ticker
                                    if "timestamp" not in item:
                                        item["timestamp"] = datetime.utcnow()
                                collection.insert_many(data_to_insert)
                                st.info("📦 News articles saved to database")
                            except Exception as db_error:
                                st.warning(f"⚠️ Could not save to database: {db_error}")
                        
                        # Display source breakdown
                        st.divider()
                        st.subheader("📊 News Sources")
                        
                        if 'source' in news_df.columns:
                            source_counts = news_df['source'].value_counts()
                            source_cols = st.columns(min(len(source_counts), 4))
                            
                            for idx, (source, count) in enumerate(source_counts.items()):
                                with source_cols[idx % len(source_cols)]:
                                    st.metric(source, count)
                        
                        # ========== SENTIMENT ANALYSIS ==========
                        if 'analyzed_df' in st.session_state and not st.session_state.analyzed_df.empty:
                            analyzed_df = st.session_state.analyzed_df
                            
                            st.divider()
                            st.subheader("🎭 Sentiment Distribution")
                            
                            # Normalize sentiment labels
                            sentiment_map = {
                                'positive': 'Positive', 'POSITIVE': 'Positive',
                                'neutral': 'Neutral', 'NEUTRAL': 'Neutral',
                                'negative': 'Negative', 'NEGATIVE': 'Negative'
                            }
                            
                            analyzed_df['sentiment_normalized'] = analyzed_df['sentiment'].map(sentiment_map)
                            sentiment_counts = analyzed_df['sentiment_normalized'].value_counts()
                            
                            # Sentiment metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                positive = sentiment_counts.get('Positive', 0)
                                st.markdown(f'<div class="sentiment-positive"><h3>🟢 Positive</h3><h2>{positive}</h2></div>', unsafe_allow_html=True)
                            
                            with col2:
                                neutral = sentiment_counts.get('Neutral', 0)
                                st.markdown(f'<div class="sentiment-neutral"><h3>🟡 Neutral</h3><h2>{neutral}</h2></div>', unsafe_allow_html=True)
                            
                            with col3:
                                negative = sentiment_counts.get('Negative', 0)
                                st.markdown(f'<div class="sentiment-negative"><h3>🔴 Negative</h3><h2>{negative}</h2></div>', unsafe_allow_html=True)
                            
                            # ========== SENTIMENT PIE CHART ==========
                            st.divider()
                            
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=['Positive', 'Neutral', 'Negative'],
                                values=[positive, neutral, negative],
                                marker_colors=['#28a745', '#ffc107', '#dc3545'],
                                hole=0.4
                            )])
                            
                            fig_pie.update_layout(
                                title="Sentiment Distribution",
                                height=400
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # ========== DETAILED BREAKDOWN ==========
                            st.divider()
                            st.subheader("📋 Detailed Sentiment Breakdown")
                            
                            # Add color coding function
                            def highlight_sentiment(row):
                                if row['sentiment_value'] > 0:
                                    return ['background-color: #d4edda'] * len(row)
                                elif row['sentiment_value'] < 0:
                                    return ['background-color: #f8d7da'] * len(row)
                                else:
                                    return ['background-color: #fff3cd'] * len(row)
                            
                            display_cols = ['source', 'title', 'sentiment', 'score', 'sentiment_value']
                            available_display_cols = [col for col in display_cols if col in analyzed_df.columns]
                            
                            st.dataframe(
                                analyzed_df[available_display_cols].style.apply(highlight_sentiment, axis=1),
                                use_container_width=True,
                                height=400
                            )
                        else:
                            st.info("💡 Run 'Generate Predictions' in the **AI Predictions** tab to see detailed sentiment analysis.")
                        
            except Exception as e:
                st.error(f"❌ Error analyzing sentiment: {e}")
                st.exception(e)
        
        # Display cached sentiment if available
        elif 'analyzed_df' in st.session_state and not st.session_state.analyzed_df.empty:
            st.info("ℹ️ Showing cached sentiment analysis. Click 'Fetch Latest News' to update.")
            
            analyzed_df = st.session_state.analyzed_df
            
            # Quick sentiment tally
            sentiment_map = {
                'positive': 'Positive', 'POSITIVE': 'Positive',
                'neutral': 'Neutral', 'NEUTRAL': 'Neutral',
                'negative': 'Negative', 'NEGATIVE': 'Negative'
            }
            
            analyzed_df['sentiment_normalized'] = analyzed_df['sentiment'].map(sentiment_map)
            sentiment_counts = analyzed_df['sentiment_normalized'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🟢 Positive", sentiment_counts.get('Positive', 0))
            with col2:
                st.metric("🟡 Neutral", sentiment_counts.get('Neutral', 0))
            with col3:
                st.metric("🔴 Negative", sentiment_counts.get('Negative', 0))
            
            st.divider()
            
            display_cols = ['source', 'title', 'sentiment', 'score']
            available_cols = [col for col in display_cols if col in analyzed_df.columns]
            
            st.dataframe(
                analyzed_df[available_cols],
                use_container_width=True
            )
    
    # ========================================================================
    # TAB 4: TECHNICAL INDICATORS
    # ========================================================================
    
    with tab4:
        st.subheader(f"📈 Technical Indicators - {ticker}")
        
        if 'stock_data' in st.session_state:
            df = st.session_state.stock_data.copy()
            
            if 'Close' in df.columns and len(df) > 200:
                # Calculate moving averages
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                df['MA_200'] = df['Close'].rolling(window=200).mean()
                
                # Calculate RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # ========== MOVING AVERAGES CHART ==========
                st.subheader("📊 Moving Averages")
                
                fig_ma = go.Figure()
                
                date_col = 'Date' if 'Date' in df.columns else df.index
                
                fig_ma.add_trace(go.Scatter(x=date_col, y=df['Close'], mode='lines', name='Close', line=dict(color='#2193b0', width=2)))
                fig_ma.add_trace(go.Scatter(x=date_col, y=df['MA_20'], mode='lines', name='MA 20', line=dict(color='#ff6b6b', width=1)))
                fig_ma.add_trace(go.Scatter(x=date_col, y=df['MA_50'], mode='lines', name='MA 50', line=dict(color='#4ecdc4', width=1)))
                fig_ma.add_trace(go.Scatter(x=date_col, y=df['MA_200'], mode='lines', name='MA 200', line=dict(color='#ffe66d', width=1)))
                
                fig_ma.update_layout(
                    height=500,
                    hovermode='x unified',
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_ma, use_container_width=True)
                
                # ========== INDICATOR METRICS ==========
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📍 20-Day MA", f"${df['MA_20'].iloc[-1]:.2f}")
                with col2:
                    st.metric("📍 50-Day MA", f"${df['MA_50'].iloc[-1]:.2f}")
                with col3:
                    st.metric("📍 200-Day MA", f"${df['MA_200'].iloc[-1]:.2f}")
                with col4:
                    rsi_val = df['RSI'].iloc[-1]
                    rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                    st.metric("📊 RSI", f"{rsi_val:.1f}", rsi_status)
                
                # ========== RSI CHART ==========
                st.divider()
                st.subheader("📈 Relative Strength Index (RSI)")
                
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(x=date_col, y=df['RSI'], mode='lines', name='RSI', line=dict(color='#667eea', width=2)))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                
                fig_rsi.update_layout(
                    height=300,
                    hovermode='x unified',
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
                
            else:
                st.warning("⚠️ Insufficient data for technical indicators. Please load at least 200 days of historical data.")
        else:
            st.info("💡 Load historical data in the **Historical Data** tab first to view technical indicators.")
    
    # ========================================================================
    # TAB 5: SUMMARY REPORT
    # ========================================================================
    
    with tab5:
        st.subheader(f"📋 Comprehensive Analysis Report - {ticker}")
        
        if 'prediction' in st.session_state and 'analyzed_df' in st.session_state:
            pred = st.session_state.prediction
            analyzed_df = st.session_state.analyzed_df
            
            # ========== EXECUTIVE SUMMARY ==========
            st.markdown("### 📊 Executive Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Stock:** `{ticker}`  
                **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
                **Forecast Period:** {forecast_days} days  
                """)
            
            with col2:
                st.markdown(f"""
                **Predicted Trend:** {pred.get('trend', 'N/A')}  
                **Confidence:** {pred.get('confidence', 'N/A')}  
                **Avg Sentiment:** {pred.get('avg_sentiment', 0):+.3f}  
                """)
            
            # ========== KEY METRICS ==========
            st.divider()
            st.markdown("### 🎯 Key Predictions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Current Price**  
                {pred.get('current_price', 'N/A')}
                """)
            
            with col2:
                st.markdown(f"""
                **7-Day Target**  
                {pred.get('target_7d', 'N/A')}  
                Change: {pred.get('predicted_change_7d', 'N/A')}
                """)
            
            with col3:
                st.markdown(f"""
                **30-Day Target**  
                {pred.get('target_30d', 'N/A')}  
                Change: {pred.get('predicted_change_30d', 'N/A')}
                """)
            
            # ========== SENTIMENT SUMMARY ==========
            st.divider()
            st.markdown("### 🎭 Sentiment Analysis Summary")
            
            sentiment_map = {
                'positive': 'Positive', 'POSITIVE': 'Positive',
                'neutral': 'Neutral', 'NEUTRAL': 'Neutral',
                'negative': 'Negative', 'NEGATIVE': 'Negative'
            }
            
            analyzed_df['sentiment_normalized'] = analyzed_df['sentiment'].map(sentiment_map)
            sentiment_counts = analyzed_df['sentiment_normalized'].value_counts()
            
            total_articles = len(analyzed_df)
            positive_pct = (sentiment_counts.get('Positive', 0) / total_articles) * 100
            neutral_pct = (sentiment_counts.get('Neutral', 0) / total_articles) * 100
            negative_pct = (sentiment_counts.get('Negative', 0) / total_articles) * 100
            
            st.markdown(f"""
            **Total Articles Analyzed:** {total_articles}  
            🟢 **Positive:** {sentiment_counts.get('Positive', 0)} ({positive_pct:.1f}%)  
            🟡 **Neutral:** {sentiment_counts.get('Neutral', 0)} ({neutral_pct:.1f}%)  
            🔴 **Negative:** {sentiment_counts.get('Negative', 0)} ({negative_pct:.1f}%)  
            """)
            
            # ========== INTERPRETATION ==========
            st.divider()
            st.markdown("### 💡 Interpretation")
            
            # Generate interpretation based on sentiment and prediction
            sentiment = pred.get('avg_sentiment', 0)
            trend = pred.get('trend', '')
            
            if sentiment > 0.2 and 'Bullish' in trend:
                interpretation = "🚀 **Strong Bullish Signal**: Positive news sentiment aligns with upward price prediction."
                st.success(interpretation)
            elif sentiment < -0.2 and 'Bearish' in trend:
                interpretation = "⚠️ **Strong Bearish Signal**: Negative news sentiment aligns with downward price prediction."
                st.error(interpretation)
            elif abs(sentiment) < 0.2:
                interpretation = "➖ **Neutral Signal**: Mixed or neutral sentiment suggests sideways movement."
                st.info(interpretation)
            else:
                interpretation = "🤔 **Mixed Signals**: News sentiment and price prediction show conflicting signals. Exercise caution."
                st.warning(interpretation)
            
            # ========== DISCLAIMER ==========
            st.divider()
            st.warning("""
            ⚠️ **Important Disclaimer:**  
            This analysis is for educational and research purposes only. It should not be considered as financial advice.  
            Always conduct your own research and consult with a qualified financial advisor before making investment decisions.  
            Past performance and predictions do not guarantee future results.
            """)
            
            # ========== EXPORT OPTIONS ==========
            st.divider()
            st.markdown("### 💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download Report as JSON"):
                    report_data = {
                        'ticker': ticker,
                        'analysis_date': datetime.now().isoformat(),
                        'prediction': pred,
                        'sentiment_summary': {
                            'total_articles': total_articles,
                            'positive': int(sentiment_counts.get('Positive', 0)),
                            'neutral': int(sentiment_counts.get('Neutral', 0)),
                            'negative': int(sentiment_counts.get('Negative', 0))
                        }
                    }
                    
                    import json
                    import os
                    
                    os.makedirs("reports", exist_ok=True)
                    filename = f"reports/{ticker}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    
                    with open(filename, 'w') as f:
                        json.dump(report_data, f, indent=2, default=str)
                    
                    st.success(f"✅ Report saved to `{filename}`")
            
            with col2:
                if st.button("📊 Download Sentiment Data as CSV"):
                    import os
                    
                    os.makedirs("reports", exist_ok=True)
                    filename = f"reports/{ticker}_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    analyzed_df.to_csv(filename, index=False)
                    st.success(f"✅ Sentiment data saved to `{filename}`")
        
        else:
            st.info("💡 Run **Full Analysis** first to generate a comprehensive report.")
            st.write("The report will include:")
            st.write("• Executive summary with key metrics")
            st.write("• Detailed predictions and targets")
            st.write("• Sentiment analysis breakdown")
            st.write("• AI-generated interpretation")
            st.write("• Export options for data and reports")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>SynapseStreet</strong> · AI-Powered Stock Intelligence Platform</p>
    <p style='font-size: 0.9rem;'>Powered by FinBERT · XGBoost · NetworkX · Plotly · Streamlit</p>
    <p style='font-size: 0.8rem;'>⚠️ For educational purposes only · Not financial advice · Always do your own research</p>
    <p style='font-size: 0.7rem; margin-top: 1rem;'>© 2025 SynapseStreet · Version 2.0</p>
</div>
""", unsafe_allow_html=True)

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

