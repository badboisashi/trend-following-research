# Trend-Following Research Framework in Python

A modular backtesting and research framework for evaluating trend-following strategies across multiple assets, with realistic transaction costs, risk diagnostics, and out-of-sample evaluation.

---

## Motivation

Highly volatile assets such as cryptocurrencies often experience extreme drawdowns that make simple buy-and-hold strategies difficult to hold through time.

This project investigates whether a simple moving-average trend filter can reduce drawdowns and improve risk-adjusted returns compared to buy-and-hold, using a disciplined train/test research methodology.

The goal of the project is not to maximize returns, but to build a **robust research framework** and evaluate strategies from a **risk-aware perspective**.

---

## Research Question

**Can a simple moving-average trend filter reduce drawdowns and improve risk-adjusted returns relative to buy-and-hold, when evaluated out-of-sample?**

---

## Methodology

### Data
- Daily adjusted close prices
- Multi-asset portfolios (equally weighted by default)
- Supports equities and crypto assets (via Yahoo Finance)

### Strategy
- Binary trend-following signal based on moving averages:
  - Invested when price is above its moving average
  - Out of the market otherwise
- Buy-and-hold baseline used for comparison
- No leverage or short selling

### Execution Assumptions
- Trades executed at daily close
- Transaction costs applied on position changes
- No slippage modeling
- No leverage

---

## Train / Test Framework

- Strategy parameters (moving-average window) are selected using an **in-sample (training) period**
- The best-performing parameter is chosen based on **Sharpe ratio**
- Performance is then evaluated **out-of-sample (test period)**
- Buy-and-hold is evaluated on the same test period as a baseline

This design explicitly avoids in-sample overfitting and look-ahead bias.

---

## Risk Metrics

Performance is evaluated using the following metrics:
- Annualized volatility
- Annualized return
- Sharpe ratio (risk-free rate assumed to be zero)
- Maximum drawdown

In addition, rolling diagnostics are computed:
- Rolling volatility
- Rolling Sharpe ratio
- Drawdown curves

These diagnostics are used to understand **risk dynamics over time**, not just aggregate performance.

---

## Results Summary

Across multiple experiments, the moving-average strategy consistently reduced drawdowns relative to buy-and-hold.

In highly volatile markets such as crypto assets, trend filtering materially improved risk-adjusted returns and avoided the largest downside periods.

All reported results are evaluated strictly out-of-sample.  
No claims are made regarding future performance.

---

## Limitations

- Results are sensitive to market regimes
- No volatility targeting or leverage is applied
- Strategy universe is limited to simple trend-following rules
- No factor attribution or statistical significance testing is performed
- Transaction costs are simplified

---

## Future Extensions

Potential extensions include:
- Volatility targeting to stabilize portfolio risk across regimes
- Walk-forward optimization
- Factor exposure and attribution analysis
- Alternative momentum and trend signals

These extensions were deliberately excluded from the current version to keep the research question focused.

---


## Project Structure

```text
trend_research/
├── data.py
├── strategy.py
├── portfolio.py
├── performance.py
├── research.py
├── visualize.py
├── run.py
└── README.md

---

## How to Run

Install dependencies:
```bash 
pip install yfinance pandas numpy matplotlib