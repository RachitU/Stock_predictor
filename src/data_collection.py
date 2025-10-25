import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf

# --- Existing Stock Data Fetch Function ---
def get_stock_data(ticker: str, start='2020-01-01'):
    """Fetch historical stock data using yfinance."""
    try:
        data = yf.download(ticker, start=start, progress=False)
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None


# --- Example Placeholder (existing code) ---
def scrape_twitter(keyword: str, limit: int = 10):
    """Placeholder for Twitter scraping."""
    print(f"Scraping Twitter for {keyword} (limit={limit})...")
    return pd.DataFrame({"tweet": [f"Sample tweet about {keyword} #{i}" for i in range(limit)]})


# --- New Multi-Source News Scraper ---
def scrape_multiple_news(stock_name: str, limit: int = 10):
    """
    Scrape latest news articles related to a stock from multiple sources.
    Returns a pandas DataFrame with columns: ['source', 'title', 'link', 'snippet'].
    """

    sources = {
        "Google News": f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en",
        "Reuters": f"https://www.reuters.com/site-search/?query={stock_name}",
        "Yahoo Finance": f"https://finance.yahoo.com/quote/{stock_name}/news",
        "Economic Times": f"https://economictimes.indiatimes.com/quicksearch.cms?query={stock_name}",
    }

    articles = []

    # --- Google News (RSS Feed) ---
    try:
        r = requests.get(sources["Google News"], timeout=10)
        soup = BeautifulSoup(r.text, "xml")
        for item in soup.find_all("item")[:limit]:
            articles.append({
                "source": "Google News",
                "title": item.title.text.strip(),
                "link": item.link.text.strip(),
                "snippet": item.description.text.strip()
            })
    except Exception as e:
        print("Google News scrape failed:", e)

    # --- Yahoo Finance ---
    try:
        r = requests.get(sources["Yahoo Finance"], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("h3 a")[:limit]:
            title = a.text.strip()
            link = "https://finance.yahoo.com" + a["href"]
            articles.append({
                "source": "Yahoo Finance",
                "title": title,
                "link": link,
                "snippet": ""
            })
    except Exception as e:
        print("Yahoo scrape failed:", e)

    # --- Economic Times (India) ---
    try:
        r = requests.get(sources["Economic Times"], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select(".resultTitle a")[:limit]:
            title = a.text.strip()
            link = "https://economictimes.indiatimes.com" + a["href"]
            articles.append({
                "source": "Economic Times",
                "title": title,
                "link": link,
                "snippet": ""
            })
    except Exception as e:
        print("Economic Times scrape failed:", e)

    # --- Reuters ---
    try:
        r = requests.get(sources["Reuters"], headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.search-result-title__headline-link")[:limit]:
            title = a.text.strip()
            link = "https://www.reuters.com" + a["href"]
            articles.append({
                "source": "Reuters",
                "title": title,
                "link": link,
                "snippet": ""
            })
    except Exception as e:
        print("Reuters scrape failed:", e)

    if not articles:
        return pd.DataFrame(columns=["source", "title", "link", "snippet"])

    return pd.DataFrame(articles)
