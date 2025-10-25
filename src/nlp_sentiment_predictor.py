"""
Enhanced NLP Sentiment Predictor with ML-based Stock Movement Forecasting
Includes time-series prediction, sentiment integration, and visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.config import db, db_available, OPENAI_API_KEY
from transformers import pipeline
import openai
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import yfinance as yf
import warnings
import os # Added for example usage

warnings.filterwarnings('ignore')

# Initialize sentiment model (HuggingFace FinBERT or fallback to generic)
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    print("✅ FinBERT model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load FinBERT, using default model: {e}")
    sentiment_analyzer = pipeline("sentiment-analysis")

# Set up OpenAI (optional for prediction)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


def fetch_recent_news(ticker: str, days: int = 5):
    """Fetch recent news articles for a given stock from MongoDB."""
    if not db_available:
        print("Database not available.")
        return pd.DataFrame()

    collection = db.get_collection("news_articles")
    since = datetime.utcnow() - timedelta(days=days)
    cursor = collection.find(
        {"ticker": ticker, "timestamp": {"$gte": since}},
        {"_id": 0, "title": 1, "snippet": 1, "source": 1, "timestamp": 1}
    )

    df = pd.DataFrame(list(cursor))
    return df


def analyze_sentiment(news_df: pd.DataFrame):
    """Run sentiment analysis on news headlines/snippets."""
    if news_df.empty:
        return pd.DataFrame(), 0.0

    texts = (news_df['title'] + " " + news_df.get('snippet', "")).fillna("").tolist()
    results = sentiment_analyzer(texts)

    news_df['sentiment'] = [r['label'] for r in results]
    news_df['score'] = [r['score'] for r in results]

    # Normalize sentiment: positive = +1, neutral = 0, negative = -1
    mapping = {'positive': 1, 'neutral': 0, 'negative': -1, 'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0}
    news_df['sentiment_value'] = news_df['sentiment'].map(mapping).fillna(0)
    avg_sentiment = np.mean(news_df['sentiment_value'])
    
    return news_df, avg_sentiment


def get_historical_data(ticker: str, period: str = "6mo"):
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)
        
        if df.empty:
             # Fallback logic in case 'period' returns empty
             print(f"Warning: yfinance returned empty data for period={period}. Trying start date calculation.")
             from datetime import datetime, timedelta
             if period == '1mo': start = datetime.now() - timedelta(days=30)
             elif period == '3mo': start = datetime.now() - timedelta(days=90)
             elif period == '6mo': start = datetime.now() - timedelta(days=182)
             elif period == '1y': start = datetime.now() - timedelta(days=365)
             elif period == '2y': start = datetime.now() - timedelta(days=730)
             elif period == '5y': start = datetime.now() - timedelta(days=1825)
             else: start = datetime.now() - timedelta(days=365) # Default
             df = stock.history(start=start.strftime('%Y-%m-%d'), auto_adjust=True)

        df.reset_index(inplace=True)
        # Ensure 'Date' column is timezone-naive for comparisons
        if 'Date' in df.columns:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Remove timezone if it exists
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
            
        return df
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return pd.DataFrame()


def prepare_features(stock_df: pd.DataFrame, sentiment_value: float = 0.0):
    """Prepare features for ML model including technical indicators and sentiment."""
    df = stock_df.copy()
    
    # Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA']
    
    # Add sentiment as a feature (even if not used in model, good for records)
    df['sentiment'] = sentiment_value
    
    # Lag features (previous days' prices)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    return df


def train_prediction_model(stock_df: pd.DataFrame, sentiment_value: float = 0.0):
    """Train ML model to predict future prices."""
    df = prepare_features(stock_df, sentiment_value)
    
    # Features for prediction
    feature_cols = ['MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_signal', 
                    'volatility',
                    'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10']
    
    target_col = 'Close'
    
    model_data_cols = feature_cols + [target_col]
    valid_cols = [col for col in model_data_cols if col in df.columns]
    model_data = df[valid_cols].copy()
    
    model_data = model_data.dropna()
    
    if len(model_data) < 30:
        print(f"⚠️ Insufficient data for training. Need 30 rows, got {len(model_data)}. Try a longer lookback period.")
        return None, None, df
    
    X = model_data[feature_cols].values
    y = model_data[target_col].values
    
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
    
    model.fit(X_train, y_train)
    
    score = model.score(X_val, y_val)
    print(f"✅ Model trained - R² Score: {score:.4f}")
    
    return model, feature_cols, df


def predict_future_prices(model, last_data: pd.DataFrame, feature_cols: list, 
                          sentiment_value: float, days_ahead: int = 30):
    """
    Predict future stock prices for the next N days by re-calculating
    all features for each new predicted day.
    """
    
    temp_df = last_data.copy()
    last_date = temp_df['Date'].iloc[-1]
    predictions = []

    for day in range(days_ahead):
        df_with_features = prepare_features(temp_df, sentiment_value)
        current_features = df_with_features[feature_cols].iloc[-1].values.reshape(1, -1)
        
        pred_price = model.predict(current_features)[0]
        pred_price = max(float(pred_price), 0.0) 
        predictions.append(pred_price)
        
        new_date = last_date + timedelta(days=day + 1)
        avg_volume = temp_df['Volume'].iloc[-20:].mean() 
        
        new_row = {
            'Date': new_date,
            'Open': pred_price,
            'High': pred_price,
            'Low': pred_price,
            'Close': pred_price,
            'Volume': avg_volume
        }
        
        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

    return predictions


def create_prediction_visualization(ticker: str, historical_df: pd.DataFrame, 
                                    predictions: list, sentiment_value: float):
    """Create interactive Plotly visualization of predictions."""
    
    currency_symbol = "₹" if ticker.endswith(".NS") else "$"
    price_axis_title = f"Price ({currency_symbol})"

    hist_dates = historical_df['Date'].tolist()
    hist_prices = historical_df['Close'].tolist()
    
    last_date = historical_df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{ticker} Stock Price Prediction',
            'Prediction Confidence Bands',
            'Sentiment Impact'
        ),
        vertical_spacing=0.1,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Plot 1: Historical
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_prices,
            mode='lines',
            name='Historical Price',
            line=dict(color='#2193b0', width=2)
        ),
        row=1, col=1
    )
    
    # Plot 1: Predicted
    last_known_date = historical_df['Date'].iloc[-1]
    last_known_price = historical_df['Close'].iloc[-1]
    
    plot_future_dates = [last_known_date] + future_dates.tolist()
    plot_predictions = [last_known_price] + predictions
    
    fig.add_trace(
        go.Scatter(
            x=plot_future_dates,
            y=plot_predictions,
            mode='lines',
            name='Predicted Price',
            line=dict(color='#FFA500', width=3)
        ),
        row=1, col=1
    )
    
    # Plot 2: Confidence Bands
    upper_bound = [p * 1.05 for p in predictions]
    lower_bound = [p * 0.95 for p in predictions]
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='Confidence Band (±5%)',
            fillcolor='rgba(97, 213, 237, 0.3)'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines',
            name='Prediction',
            line=dict(color='#6dd5ed', width=2)
        ),
        row=2, col=1
    )
    
    # Plot 3: Sentiment impact
    sentiment_line = [sentiment_value] * len(predictions)
    
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=sentiment_line,
            mode='lines',
            name='Sentiment Score',
            line=dict(color='#764ba2', width=3),
            fill='tozeroy',
            fillcolor='rgba(118, 75, 162, 0.3)'
        ),
        row=3, col=1
    )
    
    fig.add_hline(y=0.2, line_dash="dot", line_color="green", 
                  annotation_text="Bullish", row=3, col=1)
    fig.add_hline(y=-0.2, line_dash="dot", line_color="red", 
                  annotation_text="Bearish", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        title=dict(
            text=f"{ticker} - AI-Powered Price Prediction with Sentiment Analysis",
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='#2193b0')
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text=price_axis_title, row=1, col=1)
    fig.update_yaxes(title_text=price_axis_title, row=2, col=1)
    fig.update_yaxes(title_text="Sentiment", row=3, col=1)
    
    return fig


def predict_stock_trend(ticker: str, avg_sentiment: float, predictions: list):
    """Enhanced prediction with ML-based forecasting."""
    
    currency_symbol = "₹" if ticker.endswith(".NS") else "$"
    
    if not predictions:
        if avg_sentiment > 0.2: trend = "📈 Likely Bullish"
        elif avg_sentiment < -0.2: trend = "📉 Likely Bearish"
        else: trend = "➖ Neutral / Sideways"
        
        return {"ticker": ticker, "avg_sentiment": avg_sentiment, "trend": trend, "timestamp": datetime.utcnow()}
    
    current_price = predictions[0]
    future_price_7d = predictions[6] if len(predictions) > 6 else predictions[-1]
    future_price_30d = predictions[-1]
    
    change_7d = ((future_price_7d - current_price) / current_price) * 100
    change_30d = ((future_price_30d - current_price) / current_price) * 100
    
    if change_30d > 5:
        trend = "📈 Strongly Bullish"
        confidence = "High" if avg_sentiment > 0.2 else "Medium"
    elif change_30d > 0:
        trend = "📈 Moderately Bullish"
        confidence = "Medium"
    elif change_30d < -5:
        trend = "📉 Strongly Bearish"
        confidence = "High" if avg_sentiment < -0.2 else "Medium"
    elif change_30d < 0:
        trend = "📉 Moderately Bearish"
        confidence = "Medium"
    else:
        trend = "➖ Neutral / Sideways"
        confidence = "Low"
    
    return {
        "ticker": ticker,
        "avg_sentiment": avg_sentiment,
        "trend": trend,
        "confidence": confidence,
        "predicted_change_7d": f"{change_7d:+.2f}%",
        "predicted_change_30d": f"{change_30d:+.2f}%",
        "current_price": f"{currency_symbol}{current_price:.2f}",
        "target_7d": f"{currency_symbol}{future_price_7d:.2f}",
        "target_30d": f"{currency_symbol}{future_price_30d:.2f}",
        "timestamp": datetime.utcnow()
    }


def nlp_sentiment_pipeline(ticker: str, forecast_days: int = 30, lookback_period: str = "6mo"):
    """
    Full pipeline: Fetch News → Sentiment Analysis → ML Prediction → Visualization
    """
    print(f"\n{'='*60}\n🧠 Starting NLP Sentiment Pipeline for {ticker}\n{'='*60}\n")
    
    # Step 1: Fetch news and analyze sentiment
    print("📰 Fetching recent news...")
    news_df = fetch_recent_news(ticker, days=5)
    
    if news_df.empty:
        print("⚠️ No recent news found. Using neutral sentiment.")
        avg_sentiment = 0.0
        analyzed_df = pd.DataFrame()
    else:
        print(f"✅ Found {len(news_df)} news articles")
        print("\n🔍 Analyzing sentiment...")
        analyzed_df, avg_sentiment = analyze_sentiment(news_df)
        print(f"✅ Average Sentiment: {avg_sentiment:.3f}")
    
    # Step 2: Get historical stock data
    print(f"\n📊 Fetching {lookback_period} of historical data for {ticker}...")
    stock_df = get_historical_data(ticker, period=lookback_period)
    
    if stock_df.empty or len(stock_df) < 30: # Added length check
        print("❌ Could not fetch sufficient stock data")
        return analyzed_df, {"error": "No stock data available"}, None
    
    print(f"✅ Retrieved {len(stock_df)} days of historical data")
    
    # Step 3: Train ML model
    print("\n🤖 Training prediction model...")
    model, feature_cols, prepared_df = train_prediction_model(stock_df, avg_sentiment)
    
    if model is None:
        print("❌ Model training failed")
        return analyzed_df, {"error": "Model training failed"}, None
    
    # Step 4: Predict future prices
    print(f"\n🔮 Predicting {forecast_days} days ahead...")
    predictions = predict_future_prices(
        model, 
        prepared_df, 
        feature_cols, 
        avg_sentiment, 
        days_ahead=forecast_days
    )
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Step 5: Create visualization
    print("\n📈 Creating visualization...")
    fig = create_prediction_visualization(ticker, stock_df, predictions, avg_sentiment)
    
    # Step 6: Generate prediction summary
    prediction = predict_stock_trend(ticker, avg_sentiment, predictions)
    
    print(f"\n{'='*60}\n📊 PREDICTION SUMMARY for {ticker}\n{'='*60}")
    print(f"Trend:              {prediction['trend']}")
    print(f"Confidence:         {prediction['confidence']}")
    print(f"Sentiment:          {avg_sentiment:+.3f}")
    print(f"7-Day Target:       {prediction['target_7d']} ({prediction['predicted_change_7d']})")
    print(f"30-Day Target:      {prediction['target_30d']} ({prediction['predicted_change_30d']})")
    print(f"{'='*60}\n")
    
    return analyzed_df, prediction, fig
if __name__ == "__main__":
    ticker = "RELIANCE.NS" # Example with Indian stock
    analyzed_df, prediction, fig = nlp_sentiment_pipeline(ticker, forecast_days=30, lookback_period="1y")
    
    if fig:
        fig.show()
        # Or save to HTML
        os.makedirs("predictions", exist_ok=True)
        fig.write_html(f"predictions/{ticker}_forecast.html")
        print(f"✅ Visualization saved to predictions/{ticker}_forecast.html")