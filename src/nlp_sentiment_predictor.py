import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.config import db, db_available, OPENAI_API_KEY
from transformers import pipeline
import openai

# ----------------------------
# Sentiment Analyzer
# ----------------------------
try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
except Exception as e:
    print("⚠️ Could not load FinBERT, using default model:", e)
    sentiment_analyzer = pipeline("sentiment-analysis")

# OpenAI API (optional)
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ----------------------------
# Fetch recent news from DB (includes Reddit)
# ----------------------------
def fetch_recent_news(ticker: str, days: int = 5):
    if not db_available:
        print("Database not available.")
        return pd.DataFrame()

    collection = db.get_collection("news_articles")
    since = datetime.utcnow() - timedelta(days=days)
    cursor = collection.find(
        {"ticker": ticker, "timestamp": {"$gte": since}},
        {"_id": 0, "title": 1, "snippet": 1, "source":1, "timestamp":1}
    )

    df = pd.DataFrame(list(cursor))
    return df

# ----------------------------
# Sentiment Analysis
# ----------------------------
def analyze_sentiment(news_df: pd.DataFrame):
    if news_df.empty:
        return pd.DataFrame(), 0.0

    texts = (news_df['title'] + " " + news_df.get('snippet', "")).fillna("").tolist()
    results = sentiment_analyzer(texts)

    news_df['sentiment'] = [r['label'] for r in results]
    news_df['score'] = [r['score'] for r in results]

    mapping = {'positive':1, 'neutral':0, 'negative':-1}
    news_df['sentiment_value'] = news_df['sentiment'].map(mapping).fillna(0)
    avg_sentiment = np.mean(news_df['sentiment_value'])
    return news_df, avg_sentiment

# ----------------------------
# Simple Rule-based Stock Prediction
# ----------------------------
def predict_stock_trend(ticker: str, avg_sentiment: float):
    if avg_sentiment > 0.2:
        trend = "📈 Likely Bullish"
    elif avg_sentiment < -0.2:
        trend = "📉 Likely Bearish"
    else:
        trend = "➖ Neutral / Sideways"

    return {
        "ticker": ticker,
        "avg_sentiment": avg_sentiment,
        "trend": trend,
        "timestamp": datetime.utcnow()
    }

# ----------------------------
# Full NLP Pipeline
# ----------------------------
def nlp_sentiment_pipeline(ticker: str):
    df = fetch_recent_news(ticker)
    if df.empty:
        return None, "No recent news or Reddit posts found."

    analyzed_df, avg_sentiment = analyze_sentiment(df)
    prediction = predict_stock_trend(ticker, avg_sentiment)
    return analyzed_df, prediction
