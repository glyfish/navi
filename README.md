# navi

A quantitative finance library for economic and financial time series analysis, statistical modeling, and strategy backtesting.

## Features

- **API clients** — async clients for [FRED](https://fred.stlouisfed.org/docs/api/fred/), [Tiingo](https://api.tiingo.com/), and [BLS](https://www.bls.gov/developers/)
- **Statistical models** — ADF, ARIMA, VAR, VECM, ECM, Brownian Motion, Fractional Brownian Motion, Ornstein-Uhlenbeck
- **Backtesting** — [backtrader](https://www.backtrader.com/)-based strategy framework with PostgreSQL persistence
- **Plots** — matplotlib visualizations for time series, forecasts, regressions, and hypothesis tests
- **MCP client** — SSE-based [Model Context Protocol](https://modelcontextprotocol.io/) client for AI tool integration

## Setup

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Description | Where to get it |
| --- | --- | --- |
| `FRED_API_KEY` | Federal Reserve Economic Data | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `BLS_API_KEY` | Bureau of Labor Statistics | [bls.gov/developers](https://data.bls.gov/registrationEngine/) |
| `TINGO_API_KEY` | Tiingo EOD prices | [tiingo.com/account/api/token](https://www.tiingo.com/account/api/token) |

Keys are loaded from `.env` at runtime via `python-dotenv`. You can also export them as shell environment variables instead.
