import pytest
import pandas as pd
import numpy as np
from data.generator import generate_mock_data
from features.factors import apply_all_factors
from nlp.sentiment import apply_sentiment_factor
from models.neutralization import neutralize_factors
from evaluation.metrics import calculate_rank_ic, calculate_quintile_spread

@pytest.fixture
def sample_data():
    # Use 50 tickers to ensure we have enough data for quintiles
    return generate_mock_data(num_tickers=50, num_days=100)

def test_factor_generation(sample_data):
    df = apply_all_factors(sample_data)
    assert 'momentum_12m_1m' in df.columns
    assert 'value_factor' in df.columns
    assert 'quality_roe' in df.columns
    
def test_sentiment_extraction(sample_data):
    df = apply_sentiment_factor(sample_data, use_mock=True)
    assert 'sentiment' in df.columns
    # Check that sentiment varies
    assert df['sentiment'].nunique() > 1
    
def test_neutralization(sample_data):
    df = apply_all_factors(sample_data)
    df['dummy_factor'] = np.random.randn(len(df))
    # Vectorized neutralization
    df_neut = neutralize_factors(df, ['dummy_factor'], category_col='sector')
    assert 'dummy_factor_neutral' in df_neut.columns
    # Check that it's actually different from raw
    assert not np.array_equal(df_neut['dummy_factor'], df_neut['dummy_factor_neutral'])
    
def test_metrics(sample_data):
    # Create a perfect correlation for testing
    sample_data['prediction'] = sample_data['fwd_1m_return']
    # Drop NAs for metric calculation
    clean_df = sample_data.dropna(subset=['prediction', 'fwd_1m_return'])
    
    ic = calculate_rank_ic(clean_df, 'prediction', 'fwd_1m_return')
    spread, df_spread = calculate_quintile_spread(clean_df, 'prediction', 'fwd_1m_return')
    
    assert isinstance(ic, float)
    assert ic > 0.5 # Should be very high given perfect correlation
    assert isinstance(spread, float)
    assert spread > 0 # High prediction should yield high return
