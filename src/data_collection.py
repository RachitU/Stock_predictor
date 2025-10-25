# # src/data_collection.py
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# from datetime import datetime, timedelta
# import feedparser
# import praw

# # --- Stock data fetch ---
# import yfinance as yf

# def get_stock_data(ticker: str, start='2020-01-01'):
#     try:
#         data = yf.download(ticker, start=start, progress=False)
#         return data
#     except Exception as e:
#         print(f"Error fetching stock data for {ticker}: {e}")
#         return None

# # --- Google News ---
# def fetch_google_news(stock_name, limit=2, max_age_days=5):
#     url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
#     cutoff = datetime.utcnow() - timedelta(days=max_age_days)
#     articles = []

#     try:
#         r = requests.get(url, timeout=10)
#         soup = BeautifulSoup(r.text, "xml")
#         for item in soup.find_all("item")[:limit*2]:  # fetch extra to filter by recency
#             pub_date = item.pubDate
#             pub_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z") if pub_date else datetime.utcnow()
#             if pub_dt < cutoff:
#                 continue
#             articles.append({
#                 "source": "Google News",
#                 "title": item.title.text.strip(),
#                 "link": item.link.text.strip(),
#                 "snippet": item.description.text.strip(),
#                 "timestamp": pub_dt
#             })
#             if len(articles) >= limit:
#                 break
#     except Exception as e:
#         print("Google News scrape failed:", e)

#     return articles

# # --- Zerodha Pulse ---
# def fetch_zerodha_pulse_news(stock_name, limit=2, max_age_days=5):
#     url = "http://pulse.zerodha.com/feed.php"
#     cutoff = datetime.utcnow() - timedelta(days=max_age_days)
#     articles = []

#     try:
#         feed = feedparser.parse(url)
#         for entry in feed.entries:
#             title_lower = entry.title.lower()
#             summary_lower = entry.summary.lower()
#             if stock_name.lower() not in title_lower and stock_name.lower() not in summary_lower:
#                 continue
#             published = datetime(*entry.published_parsed[:6])
#             if published < cutoff:
#                 continue
#             articles.append({
#                 "source": "Zerodha Pulse",
#                 "title": entry.title,
#                 "link": entry.link,
#                 "snippet": entry.summary,
#                 "timestamp": published
#             })
#             if len(articles) >= limit:
#                 break
#     except Exception as e:
#         print("Zerodha Pulse scrape failed:", e)

#     return articles

# # --- Reddit via PRAW ---
# def fetch_reddit_posts(stock_name, subreddits=["stocks", "investing"], limit=2, max_age_days=5):
#     cutoff = datetime.utcnow() - timedelta(days=max_age_days)
#     posts = []

#     try:
#         reddit = praw.Reddit(
#             client_id="YOUR_CLIENT_ID",
#             client_secret="YOUR_CLIENT_SECRET",
#             user_agent="synapsestreet-app"
#         )
#         for sub in subreddits:
#             subreddit = reddit.subreddit(sub)
#             for submission in subreddit.search(stock_name, limit=limit*2, sort="new"):
#                 created = datetime.utcfromtimestamp(submission.created_utc)
#                 if created < cutoff:
#                     continue
#                 posts.append({
#                     "source": f"Reddit/{sub}",
#                     "title": submission.title,
#                     "link": submission.url,
#                     "snippet": submission.selftext,
#                     "timestamp": created
#                 })
#                 if len(posts) >= limit:
#                     break
#     except Exception as e:
#         print("Reddit scrape failed:", e)

#     return posts

# # --- Main multi-source scraper ---
# def scrape_multiple_news(stock_name, limit=10, max_age_days=5, include_reddit=True):
#     """
#     Fetch news from multiple sources + optional Reddit.
#     Distribute articles fairly across sources.
#     """
#     articles = []

#     limit_per_source = max(1, limit // 4)  # distribute roughly equally

#     # Google News
#     articles.extend(fetch_google_news(stock_name, limit_per_source, max_age_days))

#     # Zerodha Pulse (only for NSE/Indian stocks)
#     if ".NS" in stock_name:
#         articles.extend(fetch_zerodha_pulse_news(stock_name, limit_per_source, max_age_days))

#     # Reddit
#     if include_reddit:
#         articles.extend(fetch_reddit_posts(stock_name, limit=limit_per_source, max_age_days=max_age_days))

#     # Shuffle for fair distribution
#     if articles:
#         df = pd.DataFrame(articles).sample(frac=1).reset_index(drop=True)
#         return df
#     else:
#         return pd.DataFrame(columns=["source", "title", "link", "snippet", "timestamp"])
# src/data_collection.py
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
def fetch_google_news(stock_name, limit=5, max_age_days=5):
    url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)
    articles = []

    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "xml")
        for item in soup.find_all("item")[:limit*2]:
            pub_date = item.pubDate
            pub_dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z") if pub_date else datetime.utcnow()
            if pub_dt < cutoff:
                continue
            articles.append({
                "source": "Google News",
                "title": item.title.text.strip(),
                "link": item.link.text.strip(),
                "snippet": item.description.text.strip(),
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
