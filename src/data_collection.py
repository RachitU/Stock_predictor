
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import feedparser
import yfinance as yf

# --- Stock data fetch ---
def get_stock_data(ticker: str, start='2020-01-01'):
    try:
        data = yf.download(ticker, start=start, progress=False)
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

# --- Google News ---
from dateutil import parser
from datetime import timezone

def fetch_google_news(stock_name, limit=5, max_age_days=5):
    url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)  # UTC-aware
    articles = []

    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "xml")
        for item in soup.find_all("item")[:limit*2]:
            pub_date_text = item.pubDate.text if item.pubDate else None
            try:
                pub_dt = parser.parse(pub_date_text) if pub_date_text else datetime.now(timezone.utc)
                # Ensure pub_dt is UTC-aware
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pub_dt = datetime.now(timezone.utc)

            if pub_dt < cutoff:
                continue

            articles.append({
                "source": "Google News",
                "title": item.title.text.strip() if item.title else "",
                "link": item.link.text.strip() if item.link else "",
                "snippet": item.description.text.strip() if item.description else "",
                "timestamp": pub_dt
            })

            if len(articles) >= limit:
                break

    except Exception as e:
        print("Google News scrape failed:", e)

    return articles


# --- Zerodha Pulse ---
def fetch_zerodha_pulse_news(stock_name, limit=5, max_age_days=5):
    url = "http://pulse.zerodha.com/feed.php"
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    articles = []

    # Strip .NS suffix for matching
    short_name = stock_name.replace(".NS", "").upper()

    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title_lower = entry.title.lower()
            summary_lower = entry.summary.lower()
            if short_name.lower() not in title_lower and short_name.lower() not in summary_lower:
                continue
            published = datetime(*entry.published_parsed[:6])
            if published < cutoff:
                continue
            articles.append({
                "source": "Zerodha Pulse",
                "title": entry.title,
                "link": entry.link,
                "snippet": entry.summary,
                "timestamp": published
            })
            if len(articles) >= limit:
                break
    except Exception as e:
        print("Zerodha Pulse scrape failed:", e)

    return articles

# --- Main multi-source scraper ---
def scrape_multiple_news(stock_name, limit=10, max_age_days=5):
    """
    Fetch news from multiple sources (Google News + Zerodha Pulse for Indian stocks)
    """
    articles = []
    limit_per_source = max(1, limit // 2)

    # Google News
    articles.extend(fetch_google_news(stock_name, limit_per_source, max_age_days))

    # Zerodha Pulse (Indian stocks only)
    if ".NS" in stock_name:
        articles.extend(fetch_zerodha_pulse_news(stock_name, limit_per_source, max_age_days))

    if articles:
        df = pd.DataFrame(articles).sample(frac=1).reset_index(drop=True)
        return df
    else:
        return pd.DataFrame(columns=["source", "title", "link", "snippet", "timestamp"])
