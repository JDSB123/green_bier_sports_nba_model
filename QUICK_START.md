# NBA v5.0 BETA - Quick Start Guide

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.x

## Setup (5 minutes)

### 1. Configure Environment

```powershell
# Copy template
copy .env.example .env

# Edit .env and add your API keys:
# - THE_ODDS_API_KEY
# - API_BASKETBALL_KEY
```

### 2. Start the Stack

```powershell
docker compose up -d
```

### 3. Verify Health

```powershell
curl http://localhost:8090/health
```

Expected response:
```json
{
  "status": "ok",
  "mode": "STRICT",
  "markets": 6,
  "engine_loaded": true
}
```

## Daily Usage

### Get Today's Predictions

```powershell
curl http://localhost:8090/slate/today
```

### Full Analysis with Summary Table

```powershell
python scripts/analyze_slate_docker.py --date today
```

This generates:
- Console output with picks and fire ratings
- `data/processed/slate_analysis_YYYYMMDD.txt`
- `data/processed/slate_analysis_YYYYMMDD.json`

### View Specific Game

```powershell
curl "http://localhost:8090/slate/2025-12-18"
```

## Running Backtests

```powershell
# Full pipeline (fetch + train + backtest)
docker compose -f docker-compose.backtest.yml up backtest-full

# View results
cat data/results/backtest_*.md
```

## Stopping Services

```powershell
docker compose down
```

## Next Steps

- **Full documentation:** See `README.md`
- **Architecture details:** See `docs/CURRENT_STACK_AND_FLOW.md`
- **Troubleshooting:** See `docs/DOCKER_TROUBLESHOOTING.md`
