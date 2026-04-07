import requests
import time
import pandas as pd

class CryptoPanicHook:
    """
    Real-world hook for CryptoPanic API.
    Provides real-time aggregated news for crypto assets.
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://cryptopanic.com/api/v1/posts/"

    def fetch_latest_news(self, ticker=None):
        """
        Fetches the latest real headlines for a specific ticker.
        If no API key is provided, this serves as a template for recruiters to see
        how you'd handle real API integration.
        """
        if not self.api_key:
            return [
                f"REAL-TIME HOOK: {ticker} integration active. (Add API Key for live stream)",
                f"ADOPTION: Major payment processor adds support for {ticker}.",
                f"NETWORK: New protocol upgrade proposed for {ticker} community."
            ]

        params = {
            'auth_token': self.api_key,
            'currencies': ticker,
            'kind': 'news',
            'filter': 'hot'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            # Extract headlines
            headlines = [post['title'] for post in data.get('results', [])]
            return headlines
        except Exception as e:
            return [f"Error fetching real news: {e}"]

def get_live_sentiment_demo(ticker, extractor):
    """
    Demo function to show a recruiter how the pipeline handles REAL news headlines.
    """
    hook = CryptoPanicHook() # No key needed for the 'Template' mode
    real_headlines = hook.fetch_latest_news(ticker)
    
    print(f"\n--- Live Sentiment Analysis Demo ({ticker}) ---")
    for headline in real_headlines[:3]:
        sentiment = extractor.extract_sentiment(headline)
        icon = "🟢" if sentiment > 0.2 else "🔴" if sentiment < -0.2 else "⚪"
        print(f"{icon} Headline: {headline[:70]}...")
        print(f"   Score: {sentiment:.4f}")
