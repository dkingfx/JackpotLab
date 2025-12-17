# JackpotSIM

A Powerball lottery simulator for entertainment. Generate tickets, backtest against historical draws, and experience the thrill without the risk.

## Features

- **Importance Sampling** - Weighted probability toward observed distributions
- **Frequency Analysis** - Hot/cold number tracking
- **Gap Analysis** - Overdue number detection
- **Pattern Analysis** - Even/odd, high/low distributions
- **Historical Match Detection** - Check if your numbers ever won before
- **Multiple Strategies** - Random, frequency-weighted, hybrid, and more

## Quick Start

```bash
# Install dependencies
pip install flask

# Run the web app
python app.py

# Open http://localhost:5050
```

## Strategies

| Strategy | Description |
|----------|-------------|
| Importance Sampling | P(n) = w(n)/Σw - rigorous weighted sampling |
| Frequency Weighted | Biased toward historically hot numbers |
| Overdue Weighted | Biased toward numbers due to appear |
| Hybrid | 2 hot + 2 overdue + 1 random |
| Pattern Optimized | Matches historical even/odd, sum patterns |
| Pair Based | Uses commonly co-occurring pairs |

## The Math

**Standard Monte Carlo:** P(n) = 1/69 (uniform)

**Importance Sampling:** P(n) = w(n) / Σw(i) (weighted by observed data)

This shifts probability toward observed distributions while still allowing all numbers.

## Data Source

Historical data from [New York State Open Data](https://data.ny.gov) - 1,876 draws from 2010-2025.

## Disclaimer

This is for **entertainment and educational purposes only**. Lottery outcomes are random. Past results do not influence future draws. Please gamble responsibly.

## License

MIT
