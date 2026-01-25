# NBA Basketball Prediction System (versioned by `VERSION`)

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
python scripts/predict_unified_slate.py
```

That's it. This handles everything automatically.

**Options:**
```powershell
python scripts/predict_unified_slate.py --date tomorrow        # Tomorrow's games
python scripts/predict_unified_slate.py --matchup Lakers       # Filter to Lakers
python scripts/predict_unified_slate.py --date 2025-12-19 --matchup Celtics
```

---

> **📖 Documentation:** See [`docs/`](docs/) for guides on architecture, data, deployment, and operations.

**Versioning:** `VERSION` is the canonical release identifier (current: see `VERSION`). Runtime and tooling read `VERSION` (or `NBA_MODEL_VERSION` if set), so code and deploy configs no longer hard-code versions.

## Architecture Overview

**All prediction/model computation runs in Azure Container Apps.**
Local containers and localhost endpoints are intentionally not supported to avoid conflicting environments.

### Backtests (Script-Driven)

Backtests run directly via Python scripts (no separate backtest containers).

## Quick Start

### Azure Deployment (Primary)

This repo is Azure‑first. CI/CD builds the container and deploys to Azure Container Apps.
For manual operations, see `docs/AZURE_OPERATIONS.md`.

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

## API Usage

### Get Today's Predictions

```bash
# Get current API FQDN
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/slate/today"
```

### List Enabled Markets

```bash
curl "https://$FQDN/markets"
```

### Get Comprehensive Analysis

```bash
curl "https://$FQDN/slate/today/comprehensive?use_splits=true"
```

### Single Game Prediction

```bash
curl -X POST "https://$FQDN/predict/game" \
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

Run backtests directly:

```powershell
# Production backtest (uses audited canonical data)
python scripts/historical_backtest_production.py

# Extended backtest (per-market config)
python scripts/historical_backtest_extended.py
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
python scripts/predict_unified_slate.py               # Today's games
python scripts/predict_unified_slate.py --date tomorrow
python scripts/predict_unified_slate.py --date 2025-12-18
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
└── Dockerfile.combined          # Container image used in Azure
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

- `docs/DEV_WORKFLOW.md` - Development workflow (Azure‑first)
- `docs/CURRENT_STACK_AND_FLOW.md` - Detailed architecture
- `docs/BACKTEST_QUICKSTART.md` - Backtesting guide
- `docs/DATA_INGESTION_METHODOLOGY.md` - Data sources and processing
- `docs/ARCHITECTURE.md` - Technical architecture

## Security

The system includes comprehensive security hardening:

- ✅ **Azure Container App secrets** - Production secret management
- ✅ **API Key Validation** - Fails fast if keys missing at startup
- ✅ **Optional API Authentication** - Protect endpoints with API keys
- ✅ **Circuit Breakers** - Prevent cascading API failures
- ✅ **Key Masking** - API keys never logged

**See:**
- `docs/SECURITY_HARDENING.md` - Full security guide
- `docs/AZURE_OPERATIONS.md` - Azure deployment & operations

**Quick Setup:**
```env
# Enable API authentication (optional, for production)
SERVICE_API_KEY=your_strong_random_key
REQUIRE_API_AUTH=true
```

## Troubleshooting

See `docs/AZURE_OPERATIONS.md` for common issues.

**Quick checks:**
```powershell
# Get current API FQDN
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)

# Check API health (shows API key status)
curl "https://$FQDN/health"

# View container logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg
```
