import pandas as pd
import numpy as np

def calculate_momentum(df, lookback=30, skip=1):
    """
    Crypto-native momentum: 30D return excluding the most recent day.
    """
    shifted_price_lookback = df.groupby('ticker')['close'].shift(lookback)
    shifted_price_skip = df.groupby('ticker')['close'].shift(skip)
    
    momentum = (shifted_price_skip / shifted_price_lookback) - 1
    df['momentum_30d'] = momentum
    return df

def calculate_value(df):
    """
    Crypto-native 'Value': Distance from 50-day moving average.
    A coin significantly below its MA might be 'undervalued' (mean reversion).
    """
    ma_50 = df.groupby('ticker')['close'].transform(lambda x: x.rolling(50).mean())
    df['value_factor'] = (df['close'] / ma_50) - 1
    return df

def calculate_quality(df):
    """
    Crypto-native 'Quality': Inverse of rolling volatility.
    Assets with stable returns are considered 'higher quality' for risk-adjusted alpha.
    """
    # Calculate daily returns
    df['daily_ret'] = df.groupby('ticker')['close'].pct_change()
    # Calculate rolling std
    volatility = df.groupby('ticker')['daily_ret'].transform(lambda x: x.rolling(20).std())
    df['quality_low_vol'] = 1 / volatility
    return df

def apply_all_factors(df):
    df = df.copy()
    # Sort ensure time-series calculations are correct
    df = df.sort_values(['ticker', 'date'])
    df = calculate_momentum(df)
    df = calculate_value(df)
    df = calculate_quality(df)
    return df
