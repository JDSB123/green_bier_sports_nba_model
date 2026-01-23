# Operational Runbook

**Last Updated:** 2026-01-23
**Status:** Consolidated from CONSOLIDATED_RUNBOOK, DEV_WORKFLOW, WORKFLOW_AUTOMATION

---

## The One Command

```bash
python scripts/predict_unified_slate.py
```

That's it for daily predictions.

---

## Local Development

### Prerequisites

- Python 3.11+
- Docker Desktop or Docker Engine
- Azure CLI (for deployments)

### Setup

```bash
# Clone and enter repo
cd green_bier_sports_nba_model

# Create venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows

# Install deps
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Run Scripts

```bash
# Predictions
python scripts/predict_unified_slate.py

# Run tests
pytest tests -v
```

---

## Docker (Local)

### Start API

```bash
docker compose up -d
```

**API:** http://localhost:8090

### Health Check

```bash
curl http://localhost:8090/health
```

### Test Endpoints

```bash
curl http://localhost:8090/slate/today
curl http://localhost:8090/markets
```

### Stop

```bash
docker compose down
```

---

## Backtest (Docker)

```bash
# Full pipeline
docker compose -f docker-compose.backtest.yml up backtest-full

# Data only (no backtest)
docker compose -f docker-compose.backtest.yml up backtest-data

# Interactive shell
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
```

---

## Azure Deployment

### Quick Deploy

```bash
VERSION=$(cat VERSION)

# Build
docker build --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .

# Push
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Deploy
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Verify
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health"
```

### CI/CD (Automated)

Push to `main` → GitHub Actions → Build → Deploy → Smoke test

Check status: GitHub → Actions tab

---

## Versioning

### Bump Version

```bash
python scripts/bump_version.py NBA_v33.1.9
git add VERSION models/production/model_pack.json
git commit -m "chore: bump version to NBA_v33.1.9"
git push origin main
```

### Version File

```bash
cat VERSION
# NBA_v33.1.8
```

---

## Training Models

```bash
# Train all markets
python scripts/model_train_all.py

# Validate models
python scripts/model_validate.py
```

---

## Common Tasks

| Task | Command |
|------|---------|
| Today's predictions | `python scripts/predict_unified_slate.py` |
| Tomorrow's predictions | `python scripts/predict_unified_slate.py --date tomorrow` |
| Specific team | `python scripts/predict_unified_slate.py --matchup Lakers` |
| Run tests | `pytest tests -v` |
| Validate training data | `python scripts/data_unified_validate_training.py --strict` |
| Train models | `python scripts/model_train_all.py` |
| Backtest | `python scripts/historical_backtest_production.py` |

---

## Troubleshooting

### Version check failed
```bash
python scripts/bump_version.py <VERSION>
git add . && git commit -m "fix: version sync"
```

### Missing secrets
See [SECRETS.md](SECRETS.md)

### Container health fails
```bash
docker logs nba-v33-api
curl http://localhost:8090/health
```

### Azure deployment fails
```bash
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg
```

---

## Codespaces

One-command setup:
```bash
./scripts/setup_codespace.sh
```

This:
- Creates venv and installs deps
- Syncs Codespaces secrets to `.env` and `secrets/`
- Configures git hooks

Then:
```bash
docker compose up -d
curl http://localhost:8090/health
```
