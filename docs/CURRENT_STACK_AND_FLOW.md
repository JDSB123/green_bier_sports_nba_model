# Current Stack & Ingestion Flow

**Date:** 2025-12-18  
**Status:** Docker-First Containerized Architecture

---

## Architecture Overview

**All operations run through Docker containers.** The stack consists of:

### Production (docker-compose.yml)

| Service | Port | Technology | Purpose |
|---------|------|------------|---------|
| `strict-api` | 8090 | Python/FastAPI | **Main prediction API** - 6 backtested markets |

### Backtest Services (docker-compose.backtest.yml)

| Service | Purpose |
|---------|---------|
| `backtest-full` | Fetch data + train + backtest |
| `backtest-data` | Data pipeline only |
| `backtest-only` | Backtest on existing data |
| `backtest-shell` | Interactive debugging |

---

## Performance (Backtested)

| Market | Accuracy | ROI | Predictions |
|--------|----------|-----|-------------|
| FG Spread | 60.6% | +15.7% | 422 |
| FG Total | 59.2% | +13.1% | 422 |
| FG Moneyline | 65.5% | +25.1% | 316 |
| 1H Spread | 55.9% | +8.2% | 300+ |
| 1H Total | 58.1% | +11.4% | 300+ |
| 1H Moneyline | 63.0% | +19.8% | 234 |

*Backtest period: Oct 2 - Dec 16, 2025*

---

## Running the Stack

### Start All Services

```powershell
docker compose up -d
```

### Check Health

```powershell
curl http://localhost:8090/health
```

### Get Predictions

```powershell
# Today's slate
curl http://localhost:8090/slate/today

# Comprehensive analysis with betting splits
curl "http://localhost:8090/slate/today/comprehensive?use_splits=true"
```

### Full Analysis (with summary table)

```powershell
python scripts/run_slate.py --date today
```

This script:
- Starts the production container (if needed)
- Connects to the running container API
- Fetches comprehensive analysis from the API
- Generates fire ratings and summary table
- Saves reports to `data/processed/`

---

## Running Backtests

### Full Backtest Pipeline

```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

This:
1. Validates API keys
2. Fetches fresh data from APIs
3. Builds training dataset
4. Runs backtest on all markets
5. Outputs results to `data/results/`

### Other Backtest Options

```powershell
# Data pipeline only (no backtest)
docker compose -f docker-compose.backtest.yml up backtest-data

# Backtest only (existing data)
docker compose -f docker-compose.backtest.yml up backtest-only

# Interactive shell
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
```

### Configuration

Set in `.env`:
```env
SEASONS=2024-2025,2025-2026
MARKETS=all
MIN_TRAINING=80
```

---

## Data Sources

| Source | Module | Purpose |
|--------|--------|---------|
| The Odds API | `src/ingestion/the_odds.py` | Live odds, spreads, totals, moneyline |
| API-Basketball | `src/ingestion/api_basketball.py` | Scores, stats, standings |
| ESPN | `src/ingestion/injuries.py` | Injury reports |
| Betting Splits | `src/ingestion/betting_splits.py` | Public percentages, RLM |

All data ingestion happens within containers. The `strict-api` service handles live data fetching.

---

## Data Flow (Containerized)

```
┌─────────────────────────────────────────────────────────────┐
│                   DOCKER CONTAINER                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. API Request (external)                                  │
│     curl http://localhost:8090/slate/today                  │
│                                                              │
│  2. strict-api (FastAPI)                                    │
│     └─→ Fetches live odds (The Odds API)                   │
│     └─→ Fetches betting splits                              │
│     └─→ Builds features (RichFeatureBuilder)               │
│     └─→ Runs predictions (UnifiedPredictionEngine)         │
│                                                              │
│  3. Response                                                │
│     └─→ JSON with predictions, edges, confidence           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Key API Endpoints

### `/health`
Health check with engine status.

### `/slate/{date}`
Basic predictions for a date.

### `/slate/{date}/comprehensive`
Full analysis with edges, rationale, and betting splits.

### `/predict/game` (POST)
Single game prediction with all required lines.

---

## Troubleshooting

### Container Not Running

```powershell
docker ps --filter "name=nba"
docker compose up -d
```

### Engine Not Loaded

Check logs:
```powershell
docker compose logs strict-api
```

Models may be missing from `data/processed/models/`.

### API Connection Issues

```powershell
curl http://localhost:8090/health
```

If it fails, the container isn't running or port mapping is wrong.

---

## Directory Structure

```
nba_v5.0_BETA/
├── docker-compose.yml           # Main stack
├── docker-compose.backtest.yml  # Backtest stack
├── Dockerfile                   # strict-api container
├── Dockerfile.backtest          # Backtest container
├── src/
│   ├── serving/app.py           # FastAPI main app
│   ├── prediction/              # Prediction engine
│   ├── modeling/                # Models and features
│   └── ingestion/               # Data sources
├── scripts/
│   └── analyze_slate_docker.py  # Docker-only analysis script
└── data/
    ├── processed/               # Output data (models, predictions)
    └── results/                 # Backtest results
```

---

## Summary

- ✅ **Docker-first** - Models run in containers (local runner is orchestration only)
- ✅ **6 backtested markets** - All validated with strong ROI
- ✅ **STRICT MODE** - All inputs required, no fallbacks
- ✅ **Production ready** - containerized, health checks, logging
