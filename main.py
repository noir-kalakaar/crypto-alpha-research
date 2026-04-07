import pandas as pd
import numpy as np
from data.loader import get_data
from data.news_hook import get_live_sentiment_demo
from features.factors import apply_all_factors
from nlp.sentiment import apply_sentiment_factor, SentimentExtractor
from models.neutralization import neutralize_factors
from evaluation.backtest import walk_forward_validation
import sys
import warnings

# --- USER TOGGLES ---
# Set to True to download/run the real local FinBERT model (~400MB)
USE_REAL_FINBERT = True 
# --------------------

# Suppress noisy library warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    print("=== Cross-Sectional Alpha Research: Real-World Crypto Edition ===")
    
    # 1. DATA LOADING
    print("\n1. Fetching Historical OHLCV Data via CCXT (Binance)...")
    df = get_data()
    if df.empty:
        print("Failed to fetch data.")
        return
    print(f"   Success: Loaded {df.shape[0]} daily bars for {df['ticker'].nunique()} assets.")
    
    # 2. FEATURE ENGINEERING
    print("\n2. Engineering Crypto-Native Momentum, Value, and Quality Factors...")
    df = apply_all_factors(df)
    
    # 3. NLP PIPELINE (RESEARCH MODE)
    print(f"\n3. Extracting NLP Sentiment from Simulated Research Data (USE_REAL_FINBERT={USE_REAL_FINBERT})...")
    df = apply_sentiment_factor(df, use_mock=(not USE_REAL_FINBERT))
    
    # 4. NEUTRALIZATION
    print("\n4. Neutralizing Factors against Sectors via Vectorized OLS...")
    raw_factors = ['momentum_30d', 'value_factor', 'quality_low_vol', 'sentiment']
    df = neutralize_factors(df, raw_factors, category_col='sector')
    neutralized_features = [f"{f}_neutral" for f in raw_factors]
    
    # 5. BACKTESTING
    print(f"\n5. Running Walk-Forward Validation using XGBoost on Price History...")
    try:
        results_df, rank_ic, spread = walk_forward_validation(
            df, 
            features=neutralized_features, 
            target='fwd_return', 
            train_window=100,
            step=30 
        )
        
        print("\n" + "="*50)
        print("             MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Rank IC (Spearman): {rank_ic:.4f}")
        print(f"Annualized Spread:  {spread * 100:.2f}% (Q5 vs Q1 Portfolio)")
        print("-" * 50)
        print("PIPELINE RESULT: Alpha research complete.")
        print("="*50)
            
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

    # 6. LIVE DEMO (PRODUCTION HOOK)
    # This section demonstrates real-time integration capability for recruiters
    extractor = SentimentExtractor(use_mock=(not USE_REAL_FINBERT))
    get_live_sentiment_demo("BTC", extractor)
    get_live_sentiment_demo("ETH", extractor)
    
    print("\n[COMPLETE] Your portfolio project is now ready for submission.")

if __name__ == "__main__":
    main()
