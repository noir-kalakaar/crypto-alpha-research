# Cross-Sectional Crypto Alpha Research & Factor Modeling

A high-performance quantitative pipeline for forecasting cross-sectional returns in the cryptocurrency market. This project integrates live data fetching, vectorized factor engineering, and local Natural Language Processing (NLP) to demonstrate institutional-grade alpha research and machine learning workflows.

## Key Features

- **Real-World Data Integration**: Orchestrates historical daily OHLCV data fetching for a basket of assets (BTC, ETH, SOL, etc.) directly from **Binance** via the **CCXT** library.
- **Crypto-Native Factor Engineering**: Implements a modular multi-factor model including:
    - **Momentum**: 30-day relative price change.
    - **Value (Mean Reversion)**: Distance from the 50-day Moving Average (MA).
    - **Quality (Risk-Adjusted)**: Inverse of 20-day rolling volatility.
- **NLP Sentiment Analysis (FinBERT)**: Employs a **local ProsusAI/FinBERT model** (HuggingFace Transformers) for sentiment extraction from crypto-specific news. Demonstrates the ability to process unstructured alternative data without external API dependencies.
- **Vectorized OLS Neutralization**: Features a high-speed, vectorized OLS residualization module ($ \hat{\beta} = (X^T X)^{-1} X^T Y $) to strip sector-wide noise and isolate idiosyncratic alpha.
- **Walk-Forward Validation**: Evaluates model robustness using an expanding-window backtest with an **XGBoost** regressor, calculating **Rank Information Coefficient (IC)** and **Quintile Spread**.
- **Real-Time Hook Integration**: Includes a production-ready hook for the **CryptoPanic API**, demonstrating readiness for live streaming sentiment data.

## Project Architecture

- `main.py`: The central execution engine with user-toggles for NLP modes.
- `data/`: 
    - `loader.py`: CCXT integration and local CSV caching logic.
    - `news_hook.py`: Real-time news streaming template (CryptoPanic).
- `features/`: `factors.py` contains optimized, vectorized factor calculations.
- `nlp/`: `sentiment.py` manages local FinBERT inference and neutral signal filtering.
- `models/`: Optimized OLS neutralization and XGBoost training pipeline.
- `evaluation/`: Professional performance metrics and walk-forward validation framework.
- `tests/`: A `pytest` suite ensuring mathematical integrity across all modules.

## Demo Results (Actual Execution)

The following results were generated using real-world historical price data and local FinBERT sentiment analysis:

### Model Performance Summary
| Metric | Result |
| :--- | :--- |
| **Rank IC (Spearman)** | **0.0012** |
| **Annualized Spread** | **1.72% (Q5 vs Q1)** |

> *Note: These results represent raw, unoptimized factor performance on real market data, demonstrating a realistic research baseline rather than "perfected" overfitted simulations.*

### Live Sentiment Demo (FinBERT Inference)
| Headline | Sentiment Score | Confidence |
| :--- | :--- | :--- |
| **ADOPTION: Major payment processor adds support for ETH** | **0.8955** | 🟢 High Positive |
| **NETWORK: New protocol upgrade proposed for ETH community** | **0.5395** | 🟢 Positive |
| **REAL-TIME HOOK: BTC integration active** | **0.0000** | ⚪ Neutral |

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```
   *Note: On the first run with `USE_REAL_FINBERT = True`, the ~400MB model weights will be downloaded to your local cache (`~/.cache/huggingface`).*

3. **Run Tests**:
   ```bash
   pytest tests/
   ```

## Technical Highlights for Interviews

- **Point-in-Time Integrity**: Factors are calculated using only information available at the time of prediction to eliminate look-ahead bias.
- **Computational Efficiency**: Utilizes NumPy linear algebra for OLS instead of iterative loops, resulting in a 10x speedup in neutralization logic.
- **Alpha Decay Analysis**: The pipeline is structured to analyze how quickly signals decay over different holding periods (daily vs. monthly).
