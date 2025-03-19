import logging
from datetime import datetime

import requests
from textblob import TextBlob

# Konfigurera loggning
logging.basicConfig(filename="news_analysis.log", level=logging.INFO)

# API-konfiguration
NEWS_API_URL = "https://newsapi.org/v2/everything"
API_KEY = "YOUR_NEWS_API_KEY"  # Ersätt med din API-nyckel


def fetch_news(keyword, count=10):
    """
    Hämtar de senaste nyheterna relaterade till en viss aktie eller marknad.
    """
    try:
        params = {
            "q": keyword,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": API_KEY,
        }
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        articles = [article["title"] for article in data.get("articles", [])[:count]]
        logging.info(
            f"[{datetime.now()}] ✅ Hämtade {len(articles)} nyhetsartiklar för {keyword}"
        )
        return articles
    except Exception as e:
        logging.error(
            f"[{datetime.now()}] ❌ Fel vid hämtning av nyheter för {keyword}: {str(e)}"
        )
        return []


def analyze_news_sentiment(news_articles):
    """
    Analyserar sentimentet i nyhetsrubriker med hjälp av TextBlob.
    """
    if not news_articles:
        return "neutral"

    total_polarity = sum(
        TextBlob(article).sentiment.polarity for article in news_articles
    ) / len(news_articles)
    sentiment = (
        "positivt"
        if total_polarity > 0
        else "negativt" if total_polarity < 0 else "neutral"
    )

    logging.info(
        f"[{datetime.now()}] 📊 Sentimentanalys av nyheter: {sentiment} (Polarity: {total_polarity})"
    )
    return sentiment


def fetch_and_analyze_news(keyword):
    """
    Hämtar nyhetsartiklar för en aktie och analyserar sentimentet.
    """
    news_articles = fetch_news(keyword)
    sentiment = analyze_news_sentiment(news_articles)
    return sentiment


# Exempelanrop
if __name__ == "__main__":
    stock_news_sentiment = fetch_and_analyze_news("AAPL")
    print(f"📰 Sentiment för AAPL-nyheter: {stock_news_sentiment}")
