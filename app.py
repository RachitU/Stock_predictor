"""
SynapseStreet - AI-Powered Stock Analysis with Sentiment & ML Predictions
"""

import streamlit as st
import pandas as pd
from src.config import db_available, db, OPENAI_API_KEY
from src.data_collection import get_stock_data, scrape_multiple_news
from src.nlp_sentiment_predictor import nlp_sentiment_pipeline
from datetime import datetime
import google.generativeai as genai
import os
import plotly.graph_objects as go
import yfinance as yf # Import yfinance for Company Profile

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ GEMINI_API_KEY environment variable is NOT set.")

gemini_model = genai.GenerativeModel("gemini-2.5-flash") 

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
        help="Amount of historical data to fetch (Min 6mo recommended for AI)"
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
    
    # Determine currency symbol
    currency_symbol = "₹" if ticker.endswith(".NS") else "$"
    
    # Create tabs for different views
    # 💡 ADDED A NEW TAB for Company Profile
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Historical Data",
        "🔮 AI Predictions",
        "📰 Data and Tally",
        "📈 Technical Indicators",
        "🧩 Summary Insights",
        "🏢 Company Profile"
    ])
    
    # ----------------------------
    # TAB 1: Historical Data
    # ----------------------------
    with tab1:
        st.subheader(f"Historical Stock Data - {ticker}")
        
        # Helper function to convert 'lookback_period' string to a start date
        def get_start_date_from_period(period_str):
            from datetime import datetime, timedelta
            today = datetime.now()
            
            if period_str == '1mo':
                return today - timedelta(days=30)
            elif period_str == '3mo':
                return today - timedelta(days=90)
            elif period_str == '6mo':
                return today - timedelta(days=182)
            elif period_str == '1y':
                return today - timedelta(days=365)
            elif period_str == '2y':
                return today - timedelta(days=730)
            elif period_str == '5y':
                return today - timedelta(days=1825)
            else:
                return today - timedelta(days=365) # Default to 1 year
        
        # 💡 --- FIX: SEPARATED DATA FETCHING FROM DISPLAY --- 💡
        
        # Part 1: Data Fetching (runs only when button is pressed)
        if refresh_data or run_analysis or st.button("Load Historical Data", key="load_hist"):
            try:
                start_date = get_start_date_from_period(lookback_period)
                start_date_str = start_date.strftime('%Y-%m-%d')
                
                with st.spinner(f'Fetching data since {start_date_str} for {ticker}...'):
                    df = get_stock_data(ticker, start=start_date_str)
                
                if df is None or df.empty:
                    st.error(f'Could not find stock data for ticker: **{ticker}**')
                    # Clear session state if data fetch fails
                    if 'stock_data' in st.session_state:
                        del st.session_state.stock_data
                else:
                    st.success(f'✅ Successfully fetched {len(df)} rows of data')
                    # Store in session state
                    st.session_state.stock_data = df
                    
                    # Also fetch and store profile info
                    with st.spinner("Fetching company profile..."):
                        stock_info = yf.Ticker(ticker).info
                        st.session_state.stock_info = stock_info

            except Exception as e:
                st.error(f'❌ Error fetching stock data: {e}')
                st.exception(e)
                if 'stock_data' in st.session_state:
                    del st.session_state.stock_data

        # Part 2: Data Display (runs every time if data exists in session)
        if 'stock_data' in st.session_state:
            df = st.session_state.stock_data
            
            # Handle MultiIndex columns (just in case)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.to_flat_index()
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_price = df['Close'].iloc[-1]
                st.metric("Current Price", f"{currency_symbol}{current_price:.2f}")
            with col2:
                day_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                day_change_pct = (day_change / df['Close'].iloc[-2]) * 100
                st.metric("Day Change", f"{currency_symbol}{day_change:.2f}", f"{day_change_pct:+.2f}%")
            with col3:
                period_high = df['High'].max()
                st.metric(f"Period High ({lookback_period})", f"{currency_symbol}{period_high:.2f}")
            with col4:
                avg_volume = df['Volume'].tail(20).mean()
                st.metric("Avg Volume (20D)", f"{avg_volume/1e6:.2f}M")
            
            # Data table
            st.divider()
            st.subheader(f"Data Table ({lookback_period})")
            st.dataframe(df, use_container_width=True, height=400)
            
            # Price chart
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"Price History ({lookback_period})")
            with col2:
                chart_type_tab1 = st.radio(
                    "Select Chart Type",
                    ["Line", "Candlestick"],
                    horizontal=True,
                    key="chart_toggle_tab1",
                    label_visibility="collapsed"
                )

            if chart_type_tab1 == "Line":
                st.line_chart(df['Close'], use_container_width=True)
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=ticker
                )])
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    yaxis_title=f"Price ({currency_symbol})"
                )
                st.plotly_chart(fig, use_container_width=True)
        # 💡 --- END OF FIX --- 💡
                
    # ----------------------------
    # TAB 2: AI Predictions
    # ----------------------------
    with tab2:
        st.subheader(f"🔮 AI-Powered Price Predictions - {ticker}")
        
        if run_analysis or st.button("Generate Predictions", key="gen_pred", type="primary"):
            
            if lookback_period in ['1mo', '3mo']:
                st.error("❌ Insufficient Data for AI. Please select a 'Historical Data Period' of 6mo or more to run predictions.")
            else:
                try:
                    with st.spinner(f'🧠 Running AI analysis for {ticker}...'):
                        analyzed_df, prediction, fig = nlp_sentiment_pipeline(
                            ticker, 
                            forecast_days=forecast_days, 
                            lookback_period=lookback_period
                        )
                    
                    if fig is None:
                        st.error("Could not generate predictions. Please check the ticker symbol and try again.")
                    else:
                        st.session_state.prediction = prediction
                        st.session_state.prediction_fig = fig
                        st.session_state.analyzed_df = analyzed_df
                        
                        st.success("✅ Prediction generated successfully!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Trend", prediction.get('trend', 'N/A'))
                        with col2:
                            st.metric("Sentiment Score", f"{prediction.get('avg_sentiment', 0):+.3f}")
                        with col3:
                            st.metric("7-Day Target", prediction.get('target_7d', 'N/A'), prediction.get('predicted_change_7d', 'N/A'))
                        with col4:
                            st.metric("30-Day Target", prediction.get('target_30d', 'N/A'), prediction.get('predicted_change_30d', 'N/A'))
                        
                        st.divider()
                        confidence = prediction.get('confidence', 'Unknown')
                        if confidence == 'High':
                            st.success(f"🎯 Confidence Level: **{confidence}**")
                        elif confidence == 'Medium':
                            st.info(f"📊 Confidence Level: **{confidence}**")
                        else:
                            st.warning(f"⚠️ Confidence Level: **{confidence}**")
                        
                        st.divider()
                        st.plotly_chart(fig, use_container_width=True)
                        
                        with st.expander("📋 Prediction Details"):
                            st.json(prediction)
                        
                        st.divider()
                        if st.button("💾 Save Prediction Chart"):
                            os.makedirs("predictions", exist_ok=True)
                            fig.write_html(f"predictions/{ticker}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                            st.success(f"✅ Chart saved to predictions/{ticker}_forecast.html")
                            
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {e}")
                    st.exception(e)
        
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
                    news_df = scrape_multiple_news(ticker, limit=10)
                    
                    if news_df.empty:
                        st.warning("No recent news found for this ticker.")
                    else:
                        st.success(f"✅ Found {len(news_df)} news articles")
                        
                        if db_available:
                            collection = db.get_collection("news_articles")
                            data_to_insert = news_df.to_dict(orient="records")
                            for item in data_to_insert:
                                item["ticker"] = ticker
                                item["timestamp"] = datetime.utcnow()
                            collection.insert_many(data_to_insert)
                            st.info("📦 News articles saved to database")
                        
                        if 'analyzed_df' in st.session_state and not st.session_state.analyzed_df.empty:
                            analyzed_df = st.session_state.analyzed_df
                            
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
                            
                            st.divider()
                            st.subheader("Detailed Sentiment Breakdown")
                            display_df = analyzed_df[['title', 'sentiment', 'score', 'sentiment_value', 'source']].copy()
                            
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
        
        # 💡 FIX: This display logic must also be outside the button press
        if 'stock_data' in st.session_state:
            df = st.session_state.stock_data.copy()
            
            if 'Close' in df.columns:
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                df['MA_200'] = df['Close'].rolling(window=200).mean()
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("Price vs. Moving Averages")
                with col2:
                    chart_type_tab4 = st.radio(
                        "Select Chart Type",
                        ["Line", "Candlestick"],
                        horizontal=True,
                        key="chart_toggle_tab4",
                        label_visibility="collapsed"
                    )

                chart_df = df.tail(252)

                if chart_type_tab4 == "Line":
                    st.line_chart(chart_df[['Close', 'MA_20', 'MA_50', 'MA_200']], use_container_width=True)
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=chart_df.index,
                        open=chart_df['Open'],
                        high=chart_df['High'],
                        low=chart_df['Low'],
                        close=chart_df['Close'],
                        name="Price"
                    ))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA_20'], mode='lines', name='MA 20', line=dict(color='yellow', width=1)))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA_50'], mode='lines', name='MA 50', line=dict(color='orange', width=1)))
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MA_200'], mode='lines', name='MA 200', line=dict(color='red', width=1)))
                    
                    fig.update_layout(
                        xaxis_rangeslider_visible=False,
                        yaxis_title=f"Price ({currency_symbol})",
                        legend_title="Legend"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                st.subheader("Current Indicator Values")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("20-Day MA", f"{currency_symbol}{df['MA_20'].iloc[-1]:.2f}")
                with col2:
                    st.metric("50-Day MA", f"{currency_symbol}{df['MA_50'].iloc[-1]:.2f}")
                with col3:
                    st.metric("200-Day MA", f"{currency_symbol}{df['MA_200'].iloc[-1]:.2f}")
        else:
            st.info("Load historical data on the 'Historical Data' tab to see indicators.")
    
    # ----------------------------
    # TAB 5: Gemini Summary
    # ----------------------------
    with tab5:
        st.subheader(f"🧩 AI-Powered Summary - {ticker}")

        if st.button("Generate AI Summary", key="gen_summary", type="primary", use_container_width=True):
            
            if 'prediction' not in st.session_state or 'analyzed_df' not in st.session_state or 'stock_data' not in st.session_state:
                st.warning("⚠️ Please run a 'Full Analysis' on the sidebar first to generate data for the summary.")
            
            elif not os.getenv("GEMINI_API_KEY"):
                st.error("❌ GEMINI_API_KEY is not set. Cannot generate summary.")
            
            else:
                try:
                    with st.spinner("🧠 Gemini is analyzing the data..."):
                        
                        prediction_data = st.session_state.prediction
                        sentiment_df = st.session_state.analyzed_df
                        stock_df = st.session_state.stock_data

                        current_price = stock_df['Close'].iloc[-1]
                        day_change_pct = ((stock_df['Close'].iloc[-1] / stock_df['Close'].iloc[-2]) - 1) * 100
                        period_high = stock_df['High'].max()
                        
                        trend = prediction_data.get('trend', 'N/A')
                        avg_sentiment = prediction_data.get('avg_sentiment', 0)
                        target_30d = prediction_data.get('target_30d', 'N/A')
                        change_30d = prediction_data.get('predicted_change_30d', 'N/A')
                        confidence = prediction_data.get('confidence', 'N/A')

                        sentiment_counts = sentiment_df['sentiment'].value_counts()
                        positive_count = int(sentiment_counts.get('positive', 0) + sentiment_counts.get('POSITIVE', 0))
                        neutral_count = int(sentiment_counts.get('neutral', 0) + sentiment_counts.get('NEUTRAL', 0))
                        negative_count = int(sentiment_counts.get('negative', 0) + sentiment_counts.get('NEGATIVE', 0))
                        
                        prompt = f"""
                        You are an expert financial analyst. Your task is to provide a concise, insightful summary for a retail investor based on the provided data for the stock ticker: {ticker}.
                        
                        **Do not just list the data.** You must synthesize it into a coherent narrative. Explain what the data *means* in combination.
                        
                        Here is the data:

                        **1. Current Market Data (Currency: {currency_symbol}):**
                        - Current Price: {currency_symbol}{current_price:.2f}
                        - Period High ({lookback_period}): {currency_symbol}{period_high:.2f}
                        - Today's Change: {day_change_pct:+.2f}%

                        **2. AI/ML Prediction (XGBoost Model):**
                        - Predicted Trend: {trend}
                        - 30-Day Price Target: {target_30d} (Predicted Change: {change_30d})
                        - Model Confidence: {confidence}

                        **3. News Sentiment Analysis (FinBERT Model):**
                        - Average Sentiment Score: {avg_sentiment:+.3f} (where > 0 is positive, < 0 is negative)
                        - News Article Tally: {positive_count} Positive, {neutral_count} Neutral, {negative_count} Negative
                        
                        **Your Task:**
                        Write a 3-paragraph summary covering:
                        1.  **Current Snapshot:** A brief on the stock's current price (in its correct currency) and recent performance.
                        2.  **Sentiment & News:** What is the overall market sentiment (based on the news) and how might this be influencing the stock?
                        3.  **Forward-Looking Outlook:** What does the AI prediction model suggest for the next 30 days? Combine the trend, target, and sentiment into a final concluding thought.
                        
                        Format the output as clean markdown.
                        """

                        response = gemini_model.generate_content(prompt)
                        st.markdown(response.text)
                        st.session_state.gemini_summary = response.text

                except Exception as e:
                    st.error(f"❌ Error generating Gemini summary: {e}")
                    st.exception(e)
        
        elif 'gemini_summary' in st.session_state:
            st.markdown(st.session_state.gemini_summary)
        else:
            st.info("Click the 'Generate AI Summary' button to get an AI-powered insight combining all analysis.")

    # ----------------------------
    # 💡 NEW TAB 6: Company Profile
    # ----------------------------
    with tab6:
        st.subheader(f"🏢 Company Profile")
        
        # Check if profile data was loaded in Tab 1
        if 'stock_info' not in st.session_state:
            st.info("Please click 'Load Historical Data' in Tab 1 to fetch company profile.")
        else:
            stock_info = st.session_state.stock_info
            
            st.header(stock_info.get('longName', ticker))
            
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                st.caption(f"**Industry:** {stock_info.get('industry', 'N/A')}")
            with col2:
                st.caption(f"**Website:** {stock_info.get('website', 'N/A')}")
                st.caption(f"**Country:** {stock_info.get('country', 'N/A')}")

            st.divider()
            
            st.subheader("Business Summary")
            st.markdown(stock_info.get('longBusinessSummary', 'No summary available.'))
            
            st.divider()
            
            st.subheader("Key Financials")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Cap", f"{currency_symbol}{stock_info.get('marketCap', 0)/1e9:.2f}B")
                st.metric("Enterprise Value", f"{currency_symbol}{stock_info.get('enterpriseValue', 0)/1e9:.2f}B")
            with col2:
                st.metric("Trailing P/E", f"{stock_info.get('trailingPE', 'N/A') if not isinstance(stock_info.get('trailingPE'), (int, float)) else f'{stock_info.get("trailingPE"):.2f}'}")
                st.metric("Forward P/E", f"{stock_info.get('forwardPE', 'N/A') if not isinstance(stock_info.get('forwardPE'), (int, float)) else f'{stock_info.get("forwardPE"):.2f}'}")
            with col3:
                st.metric("Dividend Yield", f"{stock_info.get('dividendYield', 0) * 100:.2f}%")
                st.metric("Beta", f"{stock_info.get('beta', 'N/A') if not isinstance(stock_info.get('beta'), (int, float)) else f'{stock_info.get("beta"):.2f}'}")
            
            st.divider()
            
            st.subheader("Analyst Recommendations")
            try:
                # Fetch recommendations on the fly for this tab
                with st.spinner("Loading analyst recommendations..."):
                    recs = yf.Ticker(ticker).recommendations
                if recs is not None and not recs.empty:
                    # Show only the last 10 recommendations
                    st.dataframe(recs.tail(10), use_container_width=True)
                else:
                    st.info("No analyst recommendations available for this stock.")
            except Exception as e:
                st.warning(f"Could not load analyst recommendations: {e}")