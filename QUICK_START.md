# NBA v5.0 BETA - Quick Start Guide

## The "Golden Path" (Single Container Production)

The production system uses a single hardened Docker container with all models baked-in. No postgres/redis/microservices required.

```powershell
./run.ps1
```

This single command:
1. Builds the production Docker image (`nba-strict-api:latest`)
2. Starts the container with baked-in models
3. Waits for the API to be healthy
4. Runs analysis and saves reports to `data/processed/`

**Options:**
```powershell
./run.ps1 --date tomorrow
./run.ps1 --matchup "Lakers"
./run.ps1 --date today --matchup "Lakers vs Celtics"
```

---

## Manual Setup (If needed)

### 1. Prerequisites
- Docker Desktop must be running
- `.env` file with API keys (copy from `.env.example` if needed)

### 2. Build and Run Container Manually
```powershell
# Build the production image
docker build -t nba-strict-api:latest -f Dockerfile .

# Run the container
docker run -d --name nba-api -p 8090:8080 --env-file .env --restart unless-stopped nba-strict-api:latest
```

### 3. Verify Health
```powershell
curl http://localhost:8090/health
```

Expected response should show `"engine_loaded": true`.

## Advanced Usage

### Full Analysis Reports (Generates Files)
`./run.ps1` already generates the detailed JSON/Text reports in `data/processed/`.
If you want to run the analysis script directly (requires container to be running):

```powershell
python scripts/analyze_slate_docker.py --date today
```

### Cleanup Old Docker Resources
Remove old NBA containers/images/volumes:
```powershell
.\scripts\cleanup_nba_docker.ps1 -All -Force
```

### Backtesting (Development)
For backtesting and model training (development only):
```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

---

## Production Model Pack

The production models are baked into the Docker image from `models/production/`:
- `spreads_model.joblib` - Full Game Spread
- `totals_model.joblib` - Full Game Total  
- `first_half_spread_model.pkl` + `first_half_spread_features.pkl` - 1H Spread
- `first_half_total_model.pkl` + `first_half_total_features.pkl` - 1H Total

All models use **logistic regression with isotonic calibration** (backtested and validated).
See `models/production/model_pack.json` for metadata and backtest results.
