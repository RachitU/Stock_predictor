import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
try:
    import snscrape.modules.twitter as sntwitter
except Exception:
    sntwitter = None

from .config import db, db_available

def scrape_twitter(keyword, limit=100, since='2024-01-01'):
    if sntwitter is None:
        raise RuntimeError('snscrape not available in this environment.')
    tweets = []
    query = f'{keyword} since:{since}'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets.append({'date': tweet.date, 'user': tweet.user.username, 'content': tweet.content})
    df = pd.DataFrame(tweets)
    if db_available and not df.empty:
        db.get_collection('tweets').insert_many(df.to_dict('records'))
    return df

def scrape_news(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = '\n'.join(p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip())
    doc = {'url': url, 'content': paragraphs}
    if db_available:
        db.get_collection('news').insert_one(doc)
    return paragraphs

def get_stock_data(ticker, start='2020-01-01'):
    df = yf.download(ticker, start=start, progress=False)
    if df is None or df.empty:
        return df
    records = df.reset_index().to_dict('records')
    if db_available:
        try:
            db.get_collection('stocks').insert_many(records)
        except Exception:
            pass
    return df
