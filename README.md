# Portfolio Optimization Dashboard

Interactive Streamlit dashboard for ETF portfolio analysis and optimization using Markowitz and Black-Litterman models.

## Features

- Historical ETF price and return analysis
- Annualized return and volatility
- Sharpe and Sortino ratios
- Value at Risk and Conditional Value at Risk
- Drawdown and watermark visualization
- Markowitz minimum-volatility portfolio
- Maximum Sharpe portfolio
- Efficient frontier
- Black-Litterman expected-return views
- Out-of-sample backtesting versus S&P 500

## Tech Stack

Python, Streamlit, pandas, NumPy, SciPy, Plotly, yfinance.

## Methodology

The app downloads adjusted ETF prices from Yahoo Finance, computes daily returns, estimates annualized return and covariance matrices, and solves constrained long-only portfolio optimization problems. The backtesting module evaluates optimized weights on a later out-of-sample period.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
