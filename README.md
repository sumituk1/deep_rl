# Deep Reinforcement Learning for Portfolio Trading

This repository contains an implementation of Deep Reinforcement Learning (DRL) for portfolio optimization using Proximal Policy Optimization (PPO).

## Overview

The project implements a PPO-based agent that learns to allocate portfolio weights across S&P 500 sector ETFs to maximize risk-adjusted returns measured by the Sharpe ratio.

## Project Structure

```
deep_rl/
├── load_data.py                 # Data loading and fetching utilities
├── networks.py                   # Neural network architectures (Policy & Value networks)
├── ppo_agent.py                  # PPO reinforcement learning agent
├── portfolio_env.py              # Portfolio trading environment
├── utils.py                      # Utility functions (Differential Sharpe, benchmarks)
├── DRL_for_portfolio_trading_chen.py  # Main training script
└── data/                         # Data directory
    └── data.pkl                  # Cached market data
```

## Key Components

### 1. **Differential Sharpe Ratio Calculator**
- Calculates instantaneous reward signals based on how returns affect the Sharpe ratio
- Uses Taylor expansion approximation for real-time feedback

### 2. **PPO Agent**
- Implements Proximal Policy Optimization with Generalized Advantage Estimation (GAE)
- Two-layer MLP networks for both policy and value functions
- Ensures portfolio weights sum to 1 using softmax activation

### 3. **Portfolio Environment**
- Simulates portfolio management with historical market data
- Features include historical returns, moving averages, and volatility metrics
- Supports cash allocation alongside equity positions

### 4. **Data Module**
- Fetches S&P 500 sector ETF data from Yahoo Finance
- Supports 11 major sector ETFs (XLK, XLF, XLV, etc.)

## Installation

```bash
pip install numpy pandas torch scikit-learn yfinance matplotlib
```

## Usage

### Basic Usage

```python
from load_data import get_sector_data
from portfolio_env import PortfolioEnvironment
from ppo_agent import PPOAgent
from utils import DifferentialSharpeCalculator

# Load data
returns_data = get_sector_data(fetch_data=False)  # Use cached data

# Create environment
env = PortfolioEnvironment(returns_data)

# Initialize agent
agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

# Train agent (see main script for full training loop)
```

### Running the Main Script

```bash
python DRL_for_portfolio_trading_chen.py
```

## Features

- **Rolling Window Training**: Trains on 3-year windows with quarterly rebalancing
- **Validation-Based Model Selection**: Uses validation Sharpe ratio for hyperparameter tuning
- **Benchmark Comparison**: Compares against equal-weight and market-cap weighted portfolios
- **Out-of-Sample Testing**: Evaluates on unseen future data

## Performance Metrics

The system evaluates performance using:
- Sharpe Ratio (annualized)
- Comparison against benchmark strategies
- Win rate vs benchmarks across multiple test periods

## References

Based on research in Deep Reinforcement Learning for portfolio management, implementing differential Sharpe ratio rewards for improved risk-adjusted returns.

## License

MIT