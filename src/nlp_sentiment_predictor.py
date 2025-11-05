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
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import yfinance as yf
import ta
import warnings
import os

warnings.filterwarnings('ignore')

# Initialize sentiment model (HuggingFace FinBERT or fallback to generic)
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    print("✅ FinBERT model loaded successfully")
except Exception as e:
    print(f"⚠ Could not load FinBERT, using default model: {e}")
    sentiment_analyzer = pipeline("sentiment-analysis")

# Set up OpenAI Client
try:
    if OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
except Exception:
    openai_client = None

# --- LLM INSIGHT GENERATION ---

def generate_llm_insight(ticker: str, recent_news_text: str, recent_reddit_posts: str, prediction: str) -> str:
    """Generates a concise textual insight based on scraped data and the prediction."""
    
    if openai_client is None:
        return "LLM Insight Not Available: OpenAI API key is missing or invalid."

    prompt = f"""
    You are a Financial Analyst AI specializing in stock forecasting.
    Analyze the following data for the company ticker symbol {ticker}:

    --- Sentiment and Prediction ---
    Current Prediction: {prediction}
    
    --- Recent News Headlines/Snippets ---
    {recent_news_text}
    
    --- Recent Reddit Discussion Summary ---
    {recent_reddit_posts}

    Based ONLY on the provided information, write a concise, one-paragraph (4-5 sentences) summary. 
    The summary must explain WHY the prediction suggests the stock movement ({prediction}) 
    by synthesizing the sentiment from the news and Reddit discussions. 
    Focus on key positive/negative drivers.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise financial analyst writing a market summary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"LLM API Error during generation: {e}"

# --- DATA FETCHING ---

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
    # 💡 Using a weighted sentiment approach (value * confidence) for better signal strength
    news_df['sentiment_value'] = news_df['sentiment'].map(mapping).fillna(0) * news_df['score']
    avg_sentiment = np.mean(news_df['sentiment_value'])

    return news_df, avg_sentiment


def get_historical_data(ticker: str, period: str = "6mo"):
    """Fetch historical stock data using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, auto_adjust=True)

        if df.empty:
             # Fallback logic in case 'period' returns empty
             from datetime import datetime, timedelta
             if period == '1mo': start = datetime.now() - timedelta(days=30)
             # Simplified fallback to ensure data is fetched
             else: start = datetime.now() - timedelta(days=365) 
             df = stock.history(start=start.strftime('%Y-%m-%d'), auto_adjust=True)

        df.reset_index(inplace=True)
        # Ensure 'Date' column is datetime and timezone-naive
        if 'Date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)

        return df
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return pd.DataFrame()


def prepare_features(stock_df: pd.DataFrame, sentiment_value: float = 0.0):
    """Prepare features for ML model including technical indicators and sentiment."""
    df = stock_df.copy()

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    # Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_width'] = bollinger.bollinger_wband()
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    if 'Volume' in df.columns:
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    else:
        df['OBV'] = 0 

    # 💡 CRITICAL CHANGE: Added Sentiment as a Feature
    df['sentiment_score'] = sentiment_value

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    # 💡 CRITICAL ACCURACY IMPROVEMENT: Define Target as Next Day's Close Price (T+1)
    df['Target_T_plus_1'] = df['Close'].shift(-1) 

    # Time Features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month

    return df


def train_prediction_model(stock_df: pd.DataFrame, sentiment_value: float = 0.0):
    """Train ML model with scaling and hyperparameter tuning."""
    df = prepare_features(stock_df, sentiment_value)

    feature_cols = [
        'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'volatility', 'BB_width', 'ATR', 'OBV',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
        'day_of_week', 'month', 'sentiment_score' # Included Sentiment as a feature
    ]

    target_col = 'Target_T_plus_1'

    model_data_cols = feature_cols + [target_col]
    valid_cols = [col for col in model_data_cols if col in df.columns]
    model_data = df[valid_cols].copy()

    # Drop rows with NaN values (removes indicator/lag NaNs and the very last row (T+1 target is NaN))
    model_data = model_data.dropna()

    if len(model_data) < 50:
        print(f"⚠ Insufficient data for training/tuning. Need 50 rows, got {len(model_data)}. Try a longer lookback period.")
        return None, None, df, None

    X = model_data[feature_cols].values
    y = model_data[target_col].values

    # Split: use last 20% for validation/testing tuning results
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 💡 SCALING: Apply Scaling only to FEATURES
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw) 

    # --- Hyperparameter Tuning with GridSearchCV ---
    print("⚙ Starting hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 250],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 7],
    }

    xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"✅ Best parameters found: {best_params}")

    # Train final model with best parameters
    final_model = XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate score on the scaled test set
    score = final_model.score(X_test_scaled, y_test)
    print(f"✅ Final Model trained - Test R² Score: {score:.4f}")

    return final_model, feature_cols, df, scaler


def predict_future_prices(model, last_data: pd.DataFrame, feature_cols: list,
                          sentiment_value: float, scaler: MinMaxScaler,
                          days_ahead: int = 30):
    """
    Predict future stock prices for the next N days by re-calculating
    all features for each new predicted day and using the trained scaler.
    """

    temp_df = last_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp_df['Date']):
         temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    
    # 💡 Ensure only one index is used for the last date
    last_date = temp_df['Date'].iloc[-1]
    predictions = []

    for day in range(days_ahead):
        # 1. Prepare features for the *current* temp_df
        # Note: We do NOT define Target_T_plus_1 here, as it's only for training
        df_with_features = prepare_features(temp_df.drop(columns=['Target_T_plus_1'], errors='ignore'), sentiment_value)

        # Get the latest features (the last row)
        current_features_raw = df_with_features[feature_cols].iloc[-1].values.reshape(1, -1)

        # Check for NaN in features before scaling
        # This is crucial because features like MACD_diff take time to fill.
        if np.isnan(current_features_raw).any():
            # Instead of using the last valid prediction, calculate an average change
            mean_change = temp_df['Close'].pct_change().mean()
            pred_price = temp_df['Close'].iloc[-1] * (1 + mean_change)
            print(f"⚠ NaN detected in features for prediction day {day+1}. Using mean price change prediction.")
        else:
            # 2. Scale features and make prediction
            current_features_scaled = scaler.transform(current_features_raw)
            pred_price = model.predict(current_features_scaled)[0]

        # 3. Append prediction and ensure positive price
        pred_price = max(float(pred_price), 0.0)
        predictions.append(pred_price)

        # 4. Create new row for the *next* day and append to temp_df
        new_date = last_date + timedelta(days=day + 1)
        avg_volume = temp_df['Volume'].iloc[-20:].mean() if 'Volume' in temp_df.columns else 0

        new_row = {
            'Date': new_date,
            'Open': pred_price, 
            'High': pred_price,
            'Low': pred_price,
            'Close': pred_price,
            'Volume': avg_volume
        }

        new_df_row = pd.DataFrame([new_row])
        cols_to_keep = [col for col in new_df_row.columns if col in temp_df.columns]
        temp_df = pd.concat([temp_df, new_df_row[cols_to_keep]], ignore_index=True)

    return predictions


# --- VISUALIZATION (UNCHANGED) ---

def create_prediction_visualization(ticker: str, historical_df: pd.DataFrame,
                                    predictions: list, sentiment_value: float):
    """Create interactive Plotly visualization of predictions."""

    currency_symbol = "₹" if ticker.endswith(".NS") else "$"
    price_axis_title = f"Price ({currency_symbol})"

    hist_dates = historical_df['Date'].tolist()
    hist_prices = historical_df['Close'].tolist()

    last_date = pd.to_datetime(historical_df['Date'].iloc[-1])
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
    upper_bound = [p * 1.05 for p in predictions] # Simple +/- 5% band
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


# --- PREDICTION SUMMARY (MODIFIED TO USE ML OUTPUT) ---
def predict_stock_trend(ticker: str, avg_sentiment: float, predictions: list):
    """Enhanced prediction with ML-based forecasting."""

    currency_symbol = "₹" if ticker.endswith(".NS") else "$"

    if not predictions:
        if avg_sentiment > 0.2: trend = "📈 Likely Bullish"
        elif avg_sentiment < -0.2: trend = "📉 Likely Bearish"
        else: trend = "➖ Neutral / Sideways"

        return {"ticker": ticker, "avg_sentiment": avg_sentiment, "trend": trend, "confidence": "Low", "timestamp": datetime.utcnow()} # Added confidence for error case

    # Use first predicted price as 'current' for change calculation base
    current_price = predictions[0]
    future_price_7d = predictions[6] if len(predictions) > 6 else predictions[-1]
    future_price_30d = predictions[-1]

    # Avoid division by zero if current_price is 0
    change_7d = ((future_price_7d - current_price) / current_price) * 100 if current_price > 0 else 0
    change_30d = ((future_price_30d - current_price) / current_price) * 100 if current_price > 0 else 0

    # Determine trend based on 30-day predicted change
    if change_30d > 5:
        trend = "📈 Strongly Bullish"
        # Confidence estimation: Higher change + strong sentiment = higher confidence
        confidence = "High" if avg_sentiment > 0.1 else "Medium"
    elif change_30d > 1: # Require > 1% change for moderately bullish
        trend = "📈 Moderately Bullish"
        confidence = "Medium"
    elif change_30d < -5:
        trend = "📉 Strongly Bearish"
        confidence = "High" if avg_sentiment < -0.1 else "Medium"
    elif change_30d < -1: # Require < -1% change for moderately bearish
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
        "current_price": f"{currency_symbol}{current_price:.2f}", # Reflects first predicted day
        "target_7d": f"{currency_symbol}{future_price_7d:.2f}",
        "target_30d": f"{currency_symbol}{future_price_30d:.2f}",
        "timestamp": datetime.utcnow()
    }


# --- FULL PIPELINE EXECUTION ---
def nlp_sentiment_pipeline(ticker: str, forecast_days: int = 30, lookback_period: str = "6mo"):
    """
    Full pipeline: Fetch News → Sentiment Analysis → ML Prediction → Visualization
    """
    print(f"\n{'='*60}\n🧠 Starting NLP Sentiment Pipeline for {ticker}\n{'='*60}\n")

    # Step 1: Fetch news and analyze sentiment
    print("📰 Fetching recent news...")
    news_df = fetch_recent_news(ticker, days=5)

    if news_df.empty:
        print("⚠ No recent news found. Using neutral sentiment (0.0).")
        avg_sentiment = 0.0
        analyzed_df = pd.DataFrame()
    else:
        print(f"✅ Found {len(news_df)} news articles")
        print("\n🔍 Analyzing sentiment...")
        analyzed_df, avg_sentiment = analyze_sentiment(news_df)
        print(f"✅ Average Sentiment: {avg_sentiment:+.3f}")

    # Step 2: Get historical stock data
    print(f"\n📊 Fetching {lookback_period} of historical data for {ticker}...")
    stock_df = get_historical_data(ticker, period=lookback_period)

    if stock_df.empty or len(stock_df) < 50:
        print("❌ Could not fetch sufficient stock data for training.")
        return analyzed_df, {"error": "Insufficient historical stock data available"}, None

    print(f"✅ Retrieved {len(stock_df)} days of historical data")

    # Step 3: Train ML model
    print("\n🤖 Training prediction model (including tuning)...")
    model, feature_cols, prepared_df, scaler = train_prediction_model(stock_df, avg_sentiment)

    if model is None or scaler is None:
        print("❌ Model training failed (likely due to insufficient data after feature engineering).")
        return analyzed_df, {"error": "Model training failed"}, None

    # Step 4: Predict future prices
    print(f"\n🔮 Predicting {forecast_days} days ahead...")
    predictions = predict_future_prices(
        model,
        prepared_df,
        feature_cols,
        avg_sentiment,
        scaler,
        days_ahead=forecast_days
    )

    if not predictions:
        print("❌ Prediction generation failed.")
        return analyzed_df, {"error": "Prediction generation failed"}, None

    print(f"✅ Generated {len(predictions)} predictions")

    # Step 5: Create visualization
    print("\n📈 Creating visualization...")
    fig = create_prediction_visualization(ticker, stock_df, predictions, avg_sentiment)

    # Step 6: Generate prediction summary
    prediction = predict_stock_trend(ticker, avg_sentiment, predictions)

    print(f"\n{'='*60}\n📊 PREDICTION SUMMARY for {ticker}\n{'='*60}")
    print(f"Trend:              {prediction['trend']}")
    print(f"Confidence:         {prediction['confidence']}")
    print(f"Sentiment:          {prediction['avg_sentiment']:+.3f}")
    print(f"7-Day Target:       {prediction['target_7d']} ({prediction['predicted_change_7d']})")
    print(f"30-Day Target:      {prediction['target_30d']} ({prediction['predicted_change_30d']})")
    print(f"{'='*60}\n")

    return analyzed_df, prediction, fig


# Example usage
if __name__ == "__main__":
    ticker = "RELIANCE.NS" # Example with Indian stock
    analyzed_df, prediction, fig = nlp_sentiment_pipeline(ticker, forecast_days=30, lookback_period="1y")

    if fig:
        fig.show()
        os.makedirs("predictions", exist_ok=True)
        fig.write_html(f"predictions/{ticker}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        print(f"✅ Visualization saved to predictions/{ticker}_forecast.html")
