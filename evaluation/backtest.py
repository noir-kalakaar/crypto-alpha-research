import pandas as pd
import numpy as np
from models.train import train_xgboost
from evaluation.metrics import calculate_rank_ic, calculate_quintile_spread

def walk_forward_validation(df, features, target='fwd_1m_return', train_window=252, step=63):
    """
    Implements a walk-forward validation (expanding window).
    """
    dates = df['date'].sort_values().unique()
    
    if len(dates) < train_window + step:
        raise ValueError("Not enough dates for backtesting.")
        
    predictions = []
    
    for i in range(train_window, len(dates) - step, step):
        train_start = dates[0]
        train_end = dates[i]
        test_start = dates[i]
        test_end = dates[i + step]
        
        train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)].dropna(subset=features + [target])
        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)].dropna(subset=features + [target])
        
        if len(train_df) == 0 or len(test_df) == 0:
            continue
            
        X_train, y_train = train_df[features], train_df[target]
        X_test = test_df[features]
        
        model = train_xgboost(X_train, y_train)
        preds = model.predict(X_test)
        
        test_df_copy = test_df.copy()
        test_df_copy['prediction'] = preds
        predictions.append(test_df_copy)
        
    if not predictions:
        return pd.DataFrame(), np.nan, np.nan
        
    all_predictions = pd.concat(predictions)
    
    rank_ic = calculate_rank_ic(all_predictions, 'prediction', target)
    spread, _ = calculate_quintile_spread(all_predictions, 'prediction', target)
    
    return all_predictions, rank_ic, spread
