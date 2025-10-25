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
from sklearn.preprocessing import MinMaxScaler # 💡 ADDED
from sklearn.model_selection import GridSearchCV # 💡 ADDED
from xgboost import XGBRegressor
import yfinance as yf
import ta # 💡 ADDED Technical Analysis library
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

    # Standard Technical indicators
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # RSI (Relative Strength Index) using ta library
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # MACD using ta library
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff() # Difference between MACD and signal line

    # Bollinger Bands using ta library
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_middle'] = bollinger.bollinger_mavg()
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    df['BB_width'] = bollinger.bollinger_wband() # Width of the bands

    # 💡 NEW: Average True Range (ATR) - Volatility measure
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

    # 💡 NEW: On-Balance Volume (OBV) - Volume/Price relationship
    # Ensure Volume exists before calculating OBV
    if 'Volume' in df.columns:
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    else:
        df['OBV'] = 0 # Assign a default value if Volume is missing


    # Add sentiment (though not used in model features currently)
    df['sentiment'] = sentiment_value

    # Lag features (previous days' prices)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    # 💡 NEW: Time Features
    df['day_of_week'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['month'] = df['Date'].dt.month

    return df


def train_prediction_model(stock_df: pd.DataFrame, sentiment_value: float = 0.0):
    """Train ML model with scaling and hyperparameter tuning."""
    df = prepare_features(stock_df, sentiment_value)

    # 💡 EXPANDED: More features including new ones
    feature_cols = [
        'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'volatility', 'BB_width', 'ATR', 'OBV',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
        'day_of_week', 'month'
    ]

    target_col = 'Close'

    model_data_cols = feature_cols + [target_col]
    valid_cols = [col for col in model_data_cols if col in df.columns]
    model_data = df[valid_cols].copy()

    # Drop rows with NaN values *after* feature calculation
    model_data = model_data.dropna()

    if len(model_data) < 50: # Increased minimum rows due to more features/tuning
        print(f"⚠️ Insufficient data for training/tuning. Need 50 rows, got {len(model_data)}. Try a longer lookback period.")
        return None, None, df, None # Return None for scaler

    X = model_data[feature_cols].values
    y = model_data[target_col].values

    # Split: use last 20% for validation/testing tuning results
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 💡 NEW: Apply Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw) # Use transform, not fit_transform on test data

    # --- 💡 NEW: Hyperparameter Tuning with GridSearchCV ---
    print("⚙️ Starting hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200], # Number of trees
        'learning_rate': [0.05, 0.1], # Step size shrinkage
        'max_depth': [3, 5], # Max depth of a tree
    }

    xgb_model = XGBRegressor(random_state=42, objective='reg:squarederror')

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3, # 3-fold cross-validation
        scoring='neg_root_mean_squared_error',
        n_jobs=-1, # Use all available CPU cores
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"✅ Best parameters found: {best_params}")

    # Train final model with best parameters on the full (scaled) training set
    final_model = XGBRegressor(**best_params, random_state=42, objective='reg:squarederror')
    final_model.fit(X_train_scaled, y_train)
    # --- End of Hyperparameter Tuning ---

    # Evaluate score on the scaled test set
    score = final_model.score(X_test_scaled, y_test) # R² score
    print(f"✅ Final Model trained - Test R² Score: {score:.4f}")

    # Return the final model, features, original df (needed for prediction loop), AND the scaler
    return final_model, feature_cols, df, scaler


def predict_future_prices(model, last_data: pd.DataFrame, feature_cols: list,
                          sentiment_value: float, scaler: MinMaxScaler, # 💡 ADDED scaler
                          days_ahead: int = 30):
    """
    Predict future stock prices for the next N days by re-calculating
    all features for each new predicted day and using the trained scaler.
    """

    temp_df = last_data.copy()
    # Ensure Date is datetime for timedelta operations
    if not pd.api.types.is_datetime64_any_dtype(temp_df['Date']):
         temp_df['Date'] = pd.to_datetime(temp_df['Date'])
    last_date = temp_df['Date'].iloc[-1]
    predictions = []

    for day in range(days_ahead):
        # 1. Prepare features for the *current* temp_df
        df_with_features = prepare_features(temp_df, sentiment_value)

        # Get the latest features (the last row)
        current_features_raw = df_with_features[feature_cols].iloc[-1].values.reshape(1, -1)

        # Check for NaN in features before scaling
        if np.isnan(current_features_raw).any():
            print(f"⚠️ NaN detected in features for prediction day {day+1}. Using last valid prediction.")
            if predictions:
                 pred_price = predictions[-1]
            else:
                 pred_price = temp_df['Close'].iloc[-1]
        else:
            # 💡 NEW: Scale the features using the trained scaler
            current_features_scaled = scaler.transform(current_features_raw)

            # 2. Make a prediction using the scaled features
            pred_price = model.predict(current_features_scaled)[0]

        # Ensure prediction isn't negative
        pred_price = max(float(pred_price), 0.0)
        predictions.append(pred_price)

        # 3. Create a new row for the *next* day
        new_date = last_date + timedelta(days=day + 1)
        # Safely calculate avg_volume, handle case where Volume might be missing
        avg_volume = temp_df['Volume'].iloc[-20:].mean() if 'Volume' in temp_df.columns else 0


        new_row = {
            'Date': new_date,
            'Open': pred_price,
            'High': pred_price, # Approximate High/Low with Close
            'Low': pred_price,
            'Close': pred_price,
            'Volume': avg_volume
        }

        # 4. Append this new_row to our temp_df
        new_df_row = pd.DataFrame([new_row])
        # Only keep columns present in the original temp_df
        cols_to_keep = [col for col in new_df_row.columns if col in temp_df.columns]
        temp_df = pd.concat([temp_df, new_df_row[cols_to_keep]], ignore_index=True)

    return predictions


def create_prediction_visualization(ticker: str, historical_df: pd.DataFrame,
                                    predictions: list, sentiment_value: float):
    """Create interactive Plotly visualization of predictions."""

    currency_symbol = "₹" if ticker.endswith(".NS") else "$"
    price_axis_title = f"Price ({currency_symbol})"

    hist_dates = historical_df['Date'].tolist()
    hist_prices = historical_df['Close'].tolist()

    # Ensure last_date is a Timestamp for date_range
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
        template='plotly_white', # Using a white template for clarity
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
        confidence = "High" if avg_sentiment > 0.1 else "Medium" # Slightly lower threshold for high confidence
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


def nlp_sentiment_pipeline(ticker: str, forecast_days: int = 30, lookback_period: str = "6mo"):
    """
    Full pipeline: Fetch News → Sentiment Analysis → ML Prediction → Visualization
    """
    print(f"\n{'='*60}\n🧠 Starting NLP Sentiment Pipeline for {ticker}\n{'='*60}\n")

    # Step 1: Fetch news and analyze sentiment
    print("📰 Fetching recent news...")
    news_df = fetch_recent_news(ticker, days=5)

    if news_df.empty:
        print("⚠️ No recent news found. Using neutral sentiment (0.0).")
        avg_sentiment = 0.0
        analyzed_df = pd.DataFrame() # Ensure analyzed_df is defined
    else:
        print(f"✅ Found {len(news_df)} news articles")
        print("\n🔍 Analyzing sentiment...")
        analyzed_df, avg_sentiment = analyze_sentiment(news_df)
        print(f"✅ Average Sentiment: {avg_sentiment:.3f}")

    # Step 2: Get historical stock data
    print(f"\n📊 Fetching {lookback_period} of historical data for {ticker}...")
    stock_df = get_historical_data(ticker, period=lookback_period)

    if stock_df.empty or len(stock_df) < 50: # Check length after fetching
        print("❌ Could not fetch sufficient stock data for training.")
        return analyzed_df, {"error": "Insufficient historical stock data available"}, None

    print(f"✅ Retrieved {len(stock_df)} days of historical data")

    # Step 3: Train ML model
    print("\n🤖 Training prediction model (including tuning)...")
    # Capture the scaler returned by the updated function
    model, feature_cols, prepared_df, scaler = train_prediction_model(stock_df, avg_sentiment)

    if model is None or scaler is None: # Check both model and scaler
        print("❌ Model training failed (likely due to insufficient data after feature engineering).")
        return analyzed_df, {"error": "Model training failed"}, None

    # Step 4: Predict future prices
    print(f"\n🔮 Predicting {forecast_days} days ahead...")
    predictions = predict_future_prices(
        model,
        prepared_df, # Use the df with all features calculated, including NaNs at the start
        feature_cols,
        avg_sentiment,
        scaler, # Pass the scaler
        days_ahead=forecast_days
    )

    if not predictions: # Check if predictions list is empty
         print("❌ Prediction generation failed.")
         return analyzed_df, {"error": "Prediction generation failed"}, None

    print(f"✅ Generated {len(predictions)} predictions")

    # Step 5: Create visualization
    print("\n📈 Creating visualization...")
    # Pass the original stock_df (before features added for training) for plotting historical actuals
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


# Example usage
if __name__ == "__main__":
    ticker = "RELIANCE.NS" # Example with Indian stock
    # Use a longer lookback period for better training with more features
    analyzed_df, prediction, fig = nlp_sentiment_pipeline(ticker, forecast_days=30, lookback_period="1y")

    if fig:
        fig.show()
        # Or save to HTML
        os.makedirs("predictions", exist_ok=True)
        fig.write_html(f"predictions/{ticker}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        print(f"✅ Visualization saved to predictions/{ticker}_forecast.html")