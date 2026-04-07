import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def calculate_rank_ic(df, pred_col, target_col='fwd_1m_return'):
    """
    Vectorized calculation of cross-sectional Rank IC.
    """
    def _spearman(sub_df):
        if len(sub_df) < 2:
            return np.nan
        # Spearman correlation is Pearson on ranks
        return sub_df[[pred_col, target_col]].corr(method='spearman').iloc[0, 1]
        
    ic_series = df.groupby('date').apply(_spearman, include_groups=False)
    return ic_series.mean()

def calculate_quintile_spread(df, pred_col, target_col='fwd_1m_return'):
    """
    Vectorized quintile spread calculation.
    """
    # Vectorized quintile assignment
    df = df.copy()
    df['quintile'] = df.groupby('date')[pred_col].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    
    # Calculate returns per quintile per date
    quintile_returns = df.groupby(['date', 'quintile'])[target_col].mean().unstack()
    
    # Spread is top quintile (4) minus bottom quintile (0)
    if 0 in quintile_returns.columns and 4 in quintile_returns.columns:
        spread = quintile_returns[4] - quintile_returns[0]
        mean_spread = spread.mean()
        # Return annualized spread (assuming ~12 monthly rebalances per year)
        return mean_spread * 12, quintile_returns
    return np.nan, quintile_returns
