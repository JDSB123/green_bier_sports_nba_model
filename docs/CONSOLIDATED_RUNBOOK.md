# NBA Model Runbook: Local → Container → Azure

**Purpose:** Single consolidated guide for developers and operators.
**Version:** NBA_v33.1.4
**Environments:** Local Development | Docker Container | Azure Production
**Last Updated:** 2026-01-21

---

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Container (Local)](#docker-container-local)
3. [Azure Deployment](#azure-deployment)
4. [Secrets & Configuration](#secrets--configuration)
5. [Troubleshooting](#troubleshooting)

---

## Local Development

### Prerequisites

```bash
# Python 3.11+
python --version

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Setup Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# THE_ODDS_API_KEY=...
# API_BASKETBALL_KEY=...
```

### Run Models Locally

```bash
# Run tests
python -m pytest tests -v

# Generate predictions
python scripts/predict.py

# Collect odds data
python scripts/run_the_odds_tomorrow.py

# Run backtest
python scripts/backtest.py
```

**Output Location:** `data/processed/` (picks, results, etc.)

---

## Docker Container (Local)

### Prerequisites

- Docker installed locally
- `.env` file with API keys (or use `secrets/` directory)
- Models trained and in `models/production/`

### Using Docker Compose (Recommended)

```bash
# Start API container
docker compose up -d

# Check logs
docker compose logs -f nba-v33-api

# Stop
docker compose down
```

**API Available At:** `http://localhost:8090`

### Health Check

```bash
curl http://localhost:8090/health

# Expected response:
# {
#   "status": "ok",
#   "version": "NBA_v33.1.4",
#   "engine_loaded": true,
#   "markets": 4,
#   "markets_list": ["1h_spread", "1h_total", "fg_spread", "fg_total"]
# }
```

### Test API Endpoints

```bash
# Get today's slate
curl http://localhost:8090/slate/today

# Get executive summary
curl http://localhost:8090/slate/today/executive

# List available markets
curl http://localhost:8090/markets
```

### Backtest in Container

```bash
# Full backtest pipeline
docker compose -f docker-compose.backtest.yml up backtest-full

# Data check only (no backtest)
docker compose -f docker-compose.backtest.yml up backtest-data

# Interactive shell
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
```

### Build Custom Image

```bash
# Read VERSION
VERSION=$(cat VERSION)

# Build (combined image for prod)
docker build \
  --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined \
  .

# Run locally
docker run -p 8090:8090 \
  -e THE_ODDS_API_KEY=$(cat secrets/THE_ODDS_API_KEY) \
  -e API_BASKETBALL_KEY=$(cat secrets/API_BASKETBALL_KEY) \
  nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

---

## Azure Deployment

### Prerequisites

- Azure CLI installed (`az --version`)
- Access to `nba-gbsv-model-rg` resource group
- Secrets already in Azure Key Vault (`nbagbs-keyvault`)

### Deploy to Azure (Step-by-Step)

#### Step 1: Ensure Code is in GitHub

```bash
git add .
git commit -m "feat: update picks logic"
git push origin main
```

#### Step 2: Build Docker Image

```bash
VERSION=$(cat VERSION)
docker build \
  --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined \
  .
```

#### Step 3: Push to Azure Container Registry (ACR)

```bash
VERSION=$(cat VERSION)
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

#### Step 4: Update Azure Container App

```bash
VERSION=$(cat VERSION)
az containerapp update \
  -n nba-gbsv-api \
  -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

#### Step 5: Verify Deployment

```bash
# Get the public URL
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv)

# Health check
curl "https://$FQDN/health"

# Test predictions
curl "https://$FQDN/slate/today"

# View logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --tail 50
```

### One-Liner Deploy

```bash
VERSION=$(cat VERSION) && \
docker build --build-arg MODEL_VERSION=$VERSION -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined . && \
az acr login -n nbagbsacr && \
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION && \
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION && \
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv) && \
curl "https://$FQDN/health"
```

### Rollback

```bash
# List recent revisions
az containerapp revision list -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query '[0:5].[name,properties.template.containers[0].image]' -o table

# Revert to previous image
az containerapp update \
  -n nba-gbsv-api \
  -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.3
```

---

## Secrets & Configuration

### Local Development

**Option A: Using `.env` File**

```bash
# Copy template
cp .env.example .env

# Edit with your keys
# THE_ODDS_API_KEY=your_key
# API_BASKETBALL_KEY=your_key

# Application reads from .env automatically
python scripts/predict.py
```

**Option B: Using Docker Secrets**

```bash
# Create secrets directory
mkdir -p secrets

# Create secret files
echo "your_odds_key" > secrets/THE_ODDS_API_KEY
echo "your_basketball_key" > secrets/API_BASKETBALL_KEY

# Docker compose mounts ./secrets → /run/secrets in container
docker compose up -d
```

### Azure Production

**Secrets stored in Azure Key Vault:** `nbagbs-keyvault`

```bash
# View secret value (if you have access)
az keyvault secret show \
  --vault-name nbagbs-keyvault \
  --name THE-ODDS-API-KEY \
  --query value -o tsv

# Update secret
az keyvault secret set \
  --vault-name nbagbs-keyvault \
  --name THE-ODDS-API-KEY \
  --value "new_key"
```

Container App automatically retrieves secrets from Key Vault at runtime.

### Environment Variables

**Key Configuration:**

| Variable | Local Default | Azure Default | Purpose |
|----------|--------------|---|---------|
| `NBA_API_PORT` | `8090` | N/A (port 8090 in compose) | API port |
| `NBA_MARKETS` | `1h_spread,1h_total,fg_spread,fg_total` | Same | Markets to run (4 only) |
| `CURRENT_SEASON` | `2025-2026` | Same | Season identifier |
| `FILTER_SPREAD_MIN_CONFIDENCE` | `0.62` | Same | Betting filter |
| `FILTER_TOTAL_MIN_CONFIDENCE` | `0.72` | Same | Betting filter |

**See `.env.example` for full list.**

---

## Troubleshooting

### Local Dev

**Issue:** `ModuleNotFoundError: No module named 'src'`

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/path/to/repo
python scripts/predict.py
```

**Issue:** API keys not found

```bash
# Check .env exists and has keys
cat .env | grep THE_ODDS_API_KEY

# Or check secrets/
ls -la secrets/
```

### Docker

**Issue:** Container fails to start

```bash
# Check logs
docker compose logs nba-v33-api

# Rebuild image
docker compose build --no-cache

# Restart
docker compose restart
```

**Issue:** Health check fails

```bash
# Check container is running
docker ps | grep nba-gbsv-api

# Manually test health endpoint
docker exec nba-gbsv-api curl http://localhost:8090/health

# Check environment inside container
docker exec nba-gbsv-api env | grep NBA
```

### Azure

**Issue:** Image not found in ACR

```bash
# List images in registry
az acr repository list -n nbagbsacr

# List tags for nba-gbsv-api
az acr repository show-tags -n nbagbsacr -r nba-gbsv-api
```

**Issue:** Container App not updating

```bash
# Check current image
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].image

# Check revision history
az containerapp revision list -n nba-gbsv-api -g nba-gbsv-model-rg

# Force pull latest
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --force-latest-revision true
```

**Issue:** Health check returns error

```bash
# View container logs in Azure
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --tail 100

# Check if models loaded
az containerapp exec -n nba-gbsv-api -g nba-gbsv-model-rg \
  --command "ls -la /app/data/processed/models/"

# Verify secrets exist in Key Vault
az keyvault secret list --vault-name nbagbs-keyvault --query '[].name' -o table
```

---

## Quick Commands

### Local

```bash
# Run tests
pytest tests -v

# Generate predictions
python scripts/predict.py

# Backtest
python scripts/backtest.py
```

### Docker

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f

# Health check
curl http://localhost:8090/health
```

### Azure

```bash
# Get FQDN
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv

# Health check
curl "https://$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)/health"

# View logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg

# Deploy new version
VERSION=$(cat VERSION) && \
docker build --build-arg MODEL_VERSION=$VERSION -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined . && \
az acr login -n nbagbsacr && \
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION && \
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

---

## Support

- **Local Issues:** Check Python environment, dependencies, and `.env` file
- **Docker Issues:** Check logs with `docker compose logs`, rebuild image
- **Azure Issues:** Check Container App logs, verify image in ACR, test secrets in Key Vault
- **Model Issues:** Verify models exist in `models/production/`, check 4-market structure
- **API Issues:** Test `/health` and `/markets` endpoints first

---

## Summary

| Environment | Start | Test | Deploy |
|-------------|-------|------|--------|
| **Local** | `python -m venv .venv; pip install -r requirements.txt` | `pytest tests -v` | Push to git (triggers Azure CI) |
| **Docker** | `docker compose up -d` | `curl http://localhost:8090/health` | Build → Push to ACR → Update Container App |
| **Azure** | N/A (always running) | `curl https://$FQDN/health` | Single command: `VERSION=$(cat VERSION) && ... az containerapp update ...` |

**Key Rule:** Always read VERSION from file, never hardcode.
