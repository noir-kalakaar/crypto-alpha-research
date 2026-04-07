import pandas as pd
import numpy as np

def generate_mock_data(num_tickers=500, num_days=252*2):
    """
    Generates synthetic equity data with realistic alpha signals embedded.
    """
    np.random.seed(42)
    tickers = [f"TICKER_{i:03d}" for i in range(num_tickers)]
    dates = pd.date_range(start="2024-01-01", periods=num_days, freq="B")
    sectors = [f"Sector_{np.random.randint(1, 11)}" for _ in range(num_tickers)]
    
    # Generate prices with drift
    prices_list = []
    for ticker in tickers:
        returns = np.random.normal(0.0002, 0.015, num_days)
        price = 100 * np.exp(np.cumsum(returns))
        prices_list.append(price)
        
    price_df = pd.DataFrame(prices_list, index=tickers, columns=dates).T
    
    # Create Panel Data (Long format)
    df = price_df.reset_index().melt(id_vars='index', var_name='ticker', value_name='price')
    df = df.rename(columns={'index': 'date'})
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Add sector mapping
    sector_map = dict(zip(tickers, sectors))
    df['sector'] = df['ticker'].map(sector_map)
    
    # Generate Factors with intentional alpha bias
    # 1. Book to Market (Value)
    df['book_to_market'] = np.random.uniform(0.1, 2.0, len(df))
    # 2. ROE (Quality)
    df['roe'] = np.random.uniform(-0.1, 0.3, len(df))
    # 3. Debt to Equity (Quality)
    df['debt_to_equity'] = np.random.uniform(0.0, 3.0, len(df))
    
    # Momentum (Simulated, will be recalculated in factors.py)
    # No signal here, it's derived from price.
    
    # 4. Sentiment (NLP) - Simulate biased transcripts
    df['is_earnings_day'] = np.random.rand(len(df)) < 0.015
    # Strong keywords for positive sentiment
    pos_keywords = ["strong", "growth", "outperformed", "dividend", "efficiency"]
    neg_keywords = ["weak", "declined", "headwinds", "challenging", "lower"]
    
    def _gen_text(is_earnings):
        if not is_earnings: return ""
        if np.random.rand() > 0.5:
            return f"The results were {np.random.choice(pos_keywords)} this quarter."
        return f"We faced some {np.random.choice(neg_keywords)} during the period."
        
    df['transcript'] = df['is_earnings_day'].apply(_gen_text)
    
    # TARGET: Forward 1M Return
    # We bake in a small signal: Forward returns = noise + (0.05 * value) + (0.05 * roe)
    # This ensures the model finds something "meaningful".
    noise = np.random.normal(0, 0.02, len(df))
    # Note: Shifted -21 for forward return logic
    # We calculate the base return from price, then add alpha bias
    df['base_fwd_return'] = df.groupby('ticker')['price'].shift(-21) / df['price'] - 1
    
    # Add Alpha Bias (only where we have forward returns)
    # High book_to_market and high ROE lead to higher future returns in this simulation
    df['fwd_1m_return'] = df['base_fwd_return'] + \
                          0.01 * df['book_to_market'] + \
                          0.01 * df['roe'] + \
                          noise
                          
    return df
