import ccxt
import pandas as pd
import numpy as np
import time
import os

def fetch_crypto_data(symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 
                              'AVAX/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'XRP/USDT'], 
                       timeframe='1d', since=None, limit=500):
    """
    Fetches historical daily OHLCV data from Binance for a list of symbols.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    all_data = []
    
    print(f"Fetching data for {len(symbols)} assets...")
    
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['ticker'] = symbol.split('/')[0]
            
            all_data.append(df)
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    full_df = pd.concat(all_data).sort_values(['ticker', 'date'])
    full_df = full_df.dropna()
    
    sector_map = {
        'BTC': 'Currency', 'ETH': 'Layer-1', 'SOL': 'Layer-1', 'ADA': 'Layer-1',
        'DOT': 'Interoperability', 'AVAX': 'Layer-1', 'LINK': 'Oracle', 
        'LTC': 'Currency', 'BCH': 'Currency', 'XRP': 'Payment'
    }
    full_df['sector'] = full_df['ticker'].map(sector_map).fillna('Other')
    
    # Calculate target (Forward 1-day return)
    full_df['fwd_return'] = full_df.groupby('ticker')['close'].shift(-1) / full_df['close'] - 1
    
    # IMPROVED: Crypto-specific simulated news for the NLP pipeline
    pos_news = [
        "Major institutional adoption announced for {ticker}.",
        "Network upgrade for {ticker} successfully implemented, increasing throughput.",
        "Positive regulatory clarity provided for {ticker} by global authorities.",
        "New integration partnership boosts utility for {ticker} ecosystem.",
        "Whale accumulation detected for {ticker} over the last 24 hours."
    ]
    neg_news = [
        "Network congestion issues reported for {ticker} mainnet.",
        "Regulatory concerns arise following new guidelines affecting {ticker}.",
        "Large-scale sell-off detected on major exchanges for {ticker}.",
        "Exploit reported in a major protocol within the {ticker} ecosystem.",
        "Market volatility leads to massive liquidations in {ticker} long positions."
    ]
    
    def _gen_news(row):
        if np.random.rand() > 0.10: # 10% chance of news per day
            return ""
        ticker = row['ticker']
        if np.random.rand() > 0.5:
            return np.random.choice(pos_news).format(ticker=ticker)
        return np.random.choice(neg_news).format(ticker=ticker)
        
    full_df['transcript'] = full_df.apply(_gen_news, axis=1)
                                     
    return full_df

def get_data(cache_file='data/crypto_cache.csv', force_refresh=False):
    if os.path.exists(cache_file) and not force_refresh:
        print(f"Loading data from cache: {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=['date'])
        return df
        
    df = fetch_crypto_data()
    if not df.empty:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        df.to_csv(cache_file, index=False)
        print(f"Data cached to {cache_file}")
    return df
