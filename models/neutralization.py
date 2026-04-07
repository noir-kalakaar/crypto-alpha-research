import pandas as pd
import numpy as np

def neutralize_factors(df, factors, category_col='sector'):
    """
    Highly optimized, vectorized factor neutralization using NumPy.
    Uses the Ordinary Least Squares (OLS) closed-form solution:
    Residual = Y - X * (X'X)^-1 * X'Y
    """
    df_out = df.copy()
    
    # Pre-process category dummies (Sectors)
    if category_col in df.columns:
        dummies = pd.get_dummies(df[category_col], drop_first=True, dtype=float)
        # Add constant (intercept) for market-neutralizing the mean
        X_full = np.hstack([np.ones((len(df), 1)), dummies.values])
    else:
        X_full = np.ones((len(df), 1))

    # We still need to process per day to maintain cross-sectional integrity
    unique_dates = df['date'].unique()
    
    for date in unique_dates:
        mask = df['date'] == date
        sub_X = X_full[mask]
        
        for factor in factors:
            Y = df.loc[mask, factor].values
            
            # Handle NaNs (common in financial data)
            valid = ~np.isnan(Y)
            if valid.sum() <= sub_X.shape[1]:
                df_out.loc[mask, f'{factor}_neutral'] = np.nan
                continue
                
            X_valid = sub_X[valid]
            Y_valid = Y[valid]
            
            # Vectorized OLS logic: beta = (X'X)^-1 X'Y
            # Using pinv (pseudo-inverse) for stability against collinearity
            try:
                beta = np.linalg.pinv(X_valid.T @ X_valid) @ X_valid.T @ Y_valid
                Y_pred = sub_X @ beta
                df_out.loc[mask, f'{factor}_neutral'] = Y - Y_pred
            except np.linalg.LinAlgError:
                df_out.loc[mask, f'{factor}_neutral'] = np.nan
                
    return df_out
