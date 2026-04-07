import numpy as np
import pandas as pd
import warnings

# Lazy import to avoid loading heavy transformers unless necessary
def get_sentiment_pipeline():
    from transformers import pipeline
    # ProsusAI/finbert is a pre-trained NLP model for financial sentiment
    # It will download to ~/.cache/huggingface on first use (~400MB)
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

class SentimentExtractor:
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        self.nlp = None
        if not self.use_mock:
            print("--- Loading FinBERT Model (Local Inference) ---")
            self.nlp = get_sentiment_pipeline()
            
    def extract_sentiment(self, text):
        if not text or pd.isna(text) or text == "":
            return 0.0
            
        if self.use_mock:
            # Smart Mock based on crypto keywords
            text_lower = text.lower()
            positive_signals = ["adoption", "upgrade", "clarity", "partnership", "accumulation", "growth"]
            negative_signals = ["congestion", "regulatory concerns", "sell-off", "exploit", "liquidations"]
            
            if any(s in text_lower for s in positive_signals):
                return np.random.uniform(0.6, 0.95)
            elif any(s in text_lower for s in negative_signals):
                return np.random.uniform(-0.95, -0.6)
            return np.random.uniform(-0.1, 0.1)
                
        else:
            try:
                # Actual FinBERT inference
                # Model returns labels: 'positive', 'negative', 'neutral'
                result = self.nlp(text[:512])[0] 
                score = result['score']
                if result['label'] == 'positive':
                    return score
                elif result['label'] == 'negative':
                    return -score
                return 0.0 # neutral
            except Exception as e:
                return 0.0

def apply_sentiment_factor(df, use_mock=True):
    extractor = SentimentExtractor(use_mock=use_mock)
    
    # Process sentiment per unique transcript to save compute
    unique_transcripts = df[df['transcript'] != ""]['transcript'].unique()
    sentiment_map = {t: extractor.extract_sentiment(t) for t in unique_transcripts}
    
    # Map back to dataframe
    df['sentiment'] = df['transcript'].map(sentiment_map).fillna(0.0)
    
    # Carry forward sentiment until the next piece of news (simulating point-in-time alpha)
    df['sentiment'] = df['sentiment'].replace(0.0, np.nan)
    df = df.sort_values(['ticker', 'date'])
    df['sentiment'] = df.groupby('ticker')['sentiment'].ffill().fillna(0.0)
    
    return df
