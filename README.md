# NBA Basketball Prediction System v33.0.11.0

Production-grade containerized system for NBA betting predictions with **4 independent markets**.

## Markets (4 Independent Models)

| Period | Spread | Total |
|--------|--------|-------|
| **1H** (First Half) | 1h_spread | 1h_total |
| **FG** (Full Game) | fg_spread | fg_total |

Each market uses independent feature engineering and model training, tailored to period-specific scoring patterns.

---

## 🚀 THE ONE COMMAND

```powershell
python scripts/run_slate.py
```

That's it. This handles everything automatically.

**Options:**
```powershell
python scripts/run_slate.py --date tomorrow        # Tomorrow's games
python scripts/run_slate.py --matchup Lakers       # Filter to Lakers
python scripts/run_slate.py --date 2025-12-19 --matchup Celtics
```

---

> **📖 Single Source of Truth:** See [`docs/STACK_FLOW_AND_VERIFICATION.md`](docs/STACK_FLOW_AND_VERIFICATION.md) for the master reference.

**Versioning:** `VERSION` is the canonical release identifier (current: see `VERSION`). Runtime and tooling read `VERSION` (or `NBA_MODEL_VERSION` if set), so code and deploy configs no longer hard-code versions.

## Architecture Overview

**All prediction/model computation runs through Docker containers.**
`scripts/run_slate.py` is a thin local *orchestrator* (starts the container and calls the API); it does not run models locally.

### Services (Docker Compose)

| Service | Port | Purpose |
|---------|------|---------|
| `nba-v33-api` | 8090 | **Main prediction API** - FastAPI with 4 independent markets (1H + FG) |

### Backtest Services (On-Demand)

| Service | Purpose |
|---------|---------|
| `backtest-full` | Full pipeline: fetch data + build training + run backtest |
| `backtest-data` | Fetch and build training data only |
| `backtest-only` | Run backtest on existing data |
| `backtest-shell` | Interactive debugging shell |

## Quick Start

### Prerequisites
## CI/CD

### CI/CD Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **GBS NBA - Build & Deploy** | Auto on `push` to main | Build Docker image + deploy to Container Apps |
| **ACR Retention** | Weekly schedule + manual | Clean up old image tags |

**Standard Flow:**
1. **Push code** → `GBS NBA - Build & Deploy` auto-triggers → builds + deploys to Container App

Quick verify (production):

```bash
# Get current API FQDN
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)

# Test health endpoint
curl "https://$FQDN/health" | jq .
```


- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.x

### Setup

**Option 1: Docker Secrets (Recommended for Production)**

1. **Create secrets from `.env` file:**
   ```powershell
   python scripts/manage_secrets.py create-from-env
   ```

2. **Or create secrets manually:**
   ```powershell
   mkdir secrets
   echo "your_key_here" > secrets\THE_ODDS_API_KEY
   echo "your_key_here" > secrets\API_BASKETBALL_KEY
   echo "your_password" > secrets\DB_PASSWORD
   ```

   Secrets are automatically mounted and take precedence over environment variables.

**Option 2: Environment Variables (Development)**

1. **Copy environment template:**
   ```powershell
   copy .env.example .env
   ```

2. **Fill in API keys in `.env`:**
   ```env
   # REQUIRED - System will not start without these
   THE_ODDS_API_KEY=your_key_here
   API_BASKETBALL_KEY=your_key_here
   
   # OPTIONAL - For production API authentication
   SERVICE_API_KEY=your_service_api_key
   REQUIRE_API_AUTH=false  # Set to true for production
   ```
   
   **Note:** The system prefers Docker secrets over `.env` files. See [`docs/DOCKER_SECRETS.md`](docs/DOCKER_SECRETS.md) for full secrets guide.

### Running Production

**Start the production container:**
```powershell
docker compose up -d
```

**Check health:**
```powershell
curl http://localhost:8090/health   # Main API
```

**View logs:**
```powershell
docker compose logs -f strict-api
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

### List Enabled Markets

```bash
curl http://localhost:8090/markets
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
    "home_ml_odds": -300,
    "away_ml_odds": 240,
    "fh_spread_line": -4.0,
    "fh_total_line": 111.0,
    "fh_home_ml_odds": -180,
    "fh_away_ml_odds": 150
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
MARKETS=all          # Or: fg_spread,fg_total,1h_spread,1h_total
MIN_TRAINING=80      # Minimum training games before predictions
```

## Analyzing Slates

Use the main prediction script to analyze slates:

```powershell
python scripts/run_slate.py               # Today's games
python scripts/run_slate.py --date tomorrow
python scripts/run_slate.py --date 2025-12-18
```

This generates comprehensive predictions and analysis for the specified date.

## Project Structure

```
NBA_main/
├── src/                         # Core Python prediction code
│   ├── ingestion/               # Data ingestion modules
│   ├── modeling/                # ML models and features
│   ├── prediction/              # Prediction engine
│   ├── tracking/                # Pick tracking and validation
│   ├── utils/                   # Utilities (security, caching, etc.)
│   └── serving/                 # FastAPI app (containerized)
├── scripts/                     # Utility scripts (see scripts/README.md)
├── tests/                       # pytest test suite
├── docs/                        # Documentation
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
| 1H Spread | 55.9% | +8.2% |
| 1H Total | 58.1% | +11.4% |

*Results from backtest: Oct 2 - Dec 16, 2025 (316+ predictions)*

## Documentation

- `docs/DEV_WORKFLOW.md` - Development workflow (Codespaces/local/backtesting)
- `docs/CURRENT_STACK_AND_FLOW.md` - Detailed architecture
- `docs/BACKTEST_QUICKSTART.md` - Backtesting guide
- `docs/DATA_INGESTION_METHODOLOGY.md` - Data sources and processing
- `docs/ARCHITECTURE.md` - Technical architecture

## Security

The system includes comprehensive security hardening:

- ✅ **Docker Secrets** - Production-grade secret management (Compose-mounted secret files)
- ✅ **API Key Validation** - Fails fast if keys missing at startup
- ✅ **Optional API Authentication** - Protect endpoints with API keys
- ✅ **Circuit Breakers** - Prevent cascading API failures
- ✅ **Key Masking** - API keys never logged
- ✅ **Docker Hardening** - Resource limits and validation

**See:**
- `docs/SECURITY_HARDENING.md` - Full security guide
- `docs/DOCKER_SECRETS.md` - Docker secrets management

**Quick Setup:**
```env
# Enable API authentication (optional, for production)
SERVICE_API_KEY=your_strong_random_key
REQUIRE_API_AUTH=true
```

## Troubleshooting

See `docs/DOCKER_TROUBLESHOOTING.md` for common issues.

**Quick checks:**
```powershell
# Ensure containers are running
docker ps --filter "name=nba"

# Check API health (shows API key status)
curl http://localhost:8090/health

# View container logs
docker compose logs -f strict-api
```

