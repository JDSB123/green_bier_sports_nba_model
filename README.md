# NBA Basketball Prediction System v5.0 BETA

Production-grade containerized system for NBA betting predictions.

> **📖 Single Source of Truth:** See [`docs/SINGLE_SOURCE_OF_TRUTH.md`](docs/SINGLE_SOURCE_OF_TRUTH.md) for the master reference.

## Architecture Overview

**All operations run through Docker containers.** No local Python execution.

### Services (Docker Compose)

| Service | Port | Purpose |
|---------|------|---------|
| `strict-api` | 8090 | **Main prediction API** - FastAPI with 6 backtested markets |
| `prediction-service` | 8082 | ML inference service |
| `api-gateway` | 8080 | Unified REST API gateway |
| `feature-store` | 8081 | Feature serving (Go) |
| `line-movement-analyzer` | 8084 | RLM detection (Go) |
| `schedule-poller` | 8085 | Game schedule aggregation (Go) |
| `postgres` | 5432 | TimescaleDB for time-series data |
| `redis` | 6379 | Caching and pub/sub |

### Backtest Services (On-Demand)

| Service | Purpose |
|---------|---------|
| `backtest-full` | Full pipeline: fetch data + build training + run backtest |
| `backtest-data` | Fetch and build training data only |
| `backtest-only` | Run backtest on existing data |
| `backtest-shell` | Interactive debugging shell |

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.x

### Setup

1. **Copy environment template:**
   ```powershell
   copy .env.example .env
   ```

2. **Fill in API keys in `.env`:**
   ```env
   THE_ODDS_API_KEY=your_key_here
   API_BASKETBALL_KEY=your_key_here
   DB_PASSWORD=nba_dev_password
   ```

### Running the Stack

**Start all services:**
```powershell
docker compose up -d
```

**Check service health:**
```powershell
curl http://localhost:8090/health   # Main API
curl http://localhost:8080/health   # Gateway
curl http://localhost:8082/health   # Prediction service
```

**View logs:**
```powershell
docker compose logs -f strict-api
docker compose logs -f prediction-service
```

**Stop all services:**
```powershell
docker compose down
```

## API Usage

### Get Today's Predictions

```bash
curl http://localhost:8090/slate/today
```

### Get Comprehensive Analysis

```bash
curl "http://localhost:8090/slate/today/comprehensive?use_splits=true"
```

### Single Game Prediction

```bash
curl -X POST http://localhost:8090/predict/game \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Cleveland Cavaliers",
    "away_team": "Chicago Bulls",
    "fg_spread_line": -7.5,
    "fg_total_line": 223.5,
    "fg_home_ml": -300,
    "fg_away_ml": 240,
    "fh_spread_line": -4.0,
    "fh_total_line": 111.0,
    "fh_home_ml": -180,
    "fh_away_ml": 150
  }'
```

## Running Backtests

All backtests run in containers:

```powershell
# Full backtest pipeline (fetch data + train + backtest)
docker compose -f docker-compose.backtest.yml up backtest-full

# Just build training data (no backtest)
docker compose -f docker-compose.backtest.yml up backtest-data

# Run backtest on existing data
docker compose -f docker-compose.backtest.yml up backtest-only

# Interactive debugging shell
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
```

### Backtest Configuration

Set these in your `.env` file:
```env
SEASONS=2024-2025,2025-2026
MARKETS=all          # Or: fg_spread,fg_total,fg_moneyline,1h_spread,1h_total
MIN_TRAINING=80      # Minimum training games before predictions
```

## Analyzing Slates (Docker Only)

Use the Docker-only analysis script:

```powershell
python scripts/analyze_slate_docker.py --date today
python scripts/analyze_slate_docker.py --date tomorrow
python scripts/analyze_slate_docker.py --date 2025-12-18
```

This script connects to the running Docker container and generates comprehensive analysis.

## Project Structure

```
nba_v5.0_BETA/
├── services/                    # Microservices (Go, Rust, Python)
│   ├── api-gateway-go/
│   ├── feature-store-go/
│   ├── line-movement-analyzer-go/
│   ├── schedule-poller-go/
│   ├── odds-ingestion-rust/
│   └── prediction-service-python/
├── src/                         # Core Python prediction code
│   ├── ingestion/               # Data ingestion modules
│   ├── modeling/                # ML models and features
│   ├── prediction/              # Prediction engine
│   └── serving/                 # FastAPI app (containerized)
├── scripts/                     # Utility scripts (run via containers)
├── database/                    # SQL migrations
├── docker-compose.yml           # Main stack
├── docker-compose.backtest.yml  # Backtest stack
├── Dockerfile                   # Main API container
└── Dockerfile.backtest          # Backtest container
```

## API Keys Required

| Key | Source | Purpose |
|-----|--------|---------|
| `THE_ODDS_API_KEY` | [The Odds API](https://the-odds-api.com/) | Live betting odds |
| `API_BASKETBALL_KEY` | [API-Sports](https://api-sports.io/) | Game outcomes, team stats |

Optional:
- `BETSAPI_KEY` - Alternative odds source
- `ACTION_NETWORK_USERNAME/PASSWORD` - Betting splits
- `KAGGLE_API_TOKEN` - Historical datasets

## Performance (Backtested)

| Market | Accuracy | ROI |
|--------|----------|-----|
| FG Spread | 60.6% | +15.7% |
| FG Total | 59.2% | +13.1% |
| FG Moneyline | 65.5% | +25.1% |
| 1H Spread | 55.9% | +8.2% |
| 1H Total | 58.1% | +11.4% |
| 1H Moneyline | 63.0% | +19.8% |

*Results from backtest: Oct 2 - Dec 16, 2025 (316+ predictions)*

## Documentation

- `docs/CURRENT_STACK_AND_FLOW.md` - Detailed architecture
- `docs/BACKTEST_QUICKSTART.md` - Backtesting guide
- `docs/DATA_INGESTION_METHODOLOGY.md` - Data sources and processing
- `docs/ARCHITECTURE.md` - Technical architecture

## Troubleshooting

See `docs/DOCKER_TROUBLESHOOTING.md` for common issues.

**Quick checks:**
```powershell
# Ensure containers are running
docker ps --filter "name=nba"

# Check API health
curl http://localhost:8090/health

# View container logs
docker compose logs -f strict-api
```
