# Azure Configuration - Single Source of Truth

**Last Updated:** 2025-12-28

## Actual Azure Architecture (Production)

```
Resource Group: nba-gbsv-model-rg
  ├─ Container Registry: nbagbsacr.azurecr.io
  ├─ Key Vault: nbagbs-keyvault
  ├─ Container Apps Environment: nba-gbsv-model-env
  │   └─ Container App: nba-gbsv-api
  │       ├─ Image: nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.8.0 (semantic) + :<git-sha> (CI)
  │       ├─ Port: 8090
  │       ├─ Scaling: 1-3 replicas
  │       └─ Environment Variables:
  │           ├─ THE_ODDS_API_KEY
  │           ├─ API_BASKETBALL_KEY
  │           ├─ SEASONS_TO_PROCESS=2025-2026
  │           └─ NBA_MODEL_VERSION=NBA_v33.0.8.0
```

## Resource Names

| Resource | Name |
|----------|------|
| Resource Group | `nba-gbsv-model-rg` |
| Container Apps Environment | `nba-gbsv-model-env` |
| Container App | `nba-gbsv-api` |
| Container Registry | `nbagbsacr` |
| Key Vault | `nbagbs-keyvault` |

## Get Current API URL (dynamic)

```bash
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv

export NBA_API_URL="https://$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)"
curl "$NBA_API_URL/health"
```

## Environment Variables (common)

| Variable | Default | Description |
|----------|---------|-------------|
| `NBA_API_URL` | `http://localhost:${NBA_API_PORT}` | Full API URL |
| `NBA_API_PORT` | `8090` | Local container port |
| `ALLOWED_ORIGINS` | `*` | CORS origins |
| `TEAMS_WEBHOOK_URL` | (required for Teams bot) | Webhook URL |

## Quick Commands

```bash
# Deploy new version (semantic tag)
pwsh ./infra/nba/deploy.ps1 -Tag NBA_v33.0.8.0

# What-if mode (preview changes)
pwsh ./infra/nba/deploy.ps1 -WhatIf

# View container app logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --follow

# Update container app image
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.8.0

# Build and push new image
docker build -t nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.8.0 -f Dockerfile.combined .
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.8.0

# Run locally with custom port
NBA_API_PORT=9000 docker compose up -d

# Check current scaling
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.scale
```

## CI/CD

**Workflows:**
- `.github/workflows/iac.yml` - Infrastructure deployment (Bicep) with what-if on PRs
- Container image builds use `Dockerfile.combined`, push to `nbagbsacr.azurecr.io/nba-gbsv-api:<sha>`

Semantic tag `NBA_v33.0.8.0` should be pushed for releases (manual or scripted).

## Model Version

- **Target:** NBA_v33.0.8.0 (4 markets: 1H + FG spreads/totals)
- **Markets:** 1h_spread, 1h_total, fg_spread, fg_total
- **Dockerfile:** `Dockerfile.combined`

## API Endpoints (core)

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check, model count, version |
| `/slate/{date}` | Picks for date (YYYY-MM-DD, today, tomorrow) |
| `/slate/{date}/comprehensive` | Full edge analysis |
| `/predict/game` | Single-game predictions |

## Important Files

| File | Purpose |
|------|---------|
| `infra/nba/main.bicep` | **Single source of truth** - ALL Azure resources |
| `infra/nba/deploy.ps1` | PowerShell wrapper for Bicep deployment |
| `infra/modules/*.bicep` | Reusable modules (storage, containerApp) |
| `.github/workflows/iac.yml` | Infrastructure CI/CD pipeline |
| `Dockerfile.combined` | Combined API + Function image |
| `azure/function_app/function_app.py` | Azure Function / Teams integration |
| `.env.example` | Environment variable template |
