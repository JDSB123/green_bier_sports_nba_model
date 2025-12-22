# Azure Configuration - SINGLE SOURCE OF TRUTH

**Last Updated:** 2025-12-22

## Actual Azure Architecture

```
NBAGBSVMODEL                               <-- Resource Group
├── nbagbsacr                              <-- Container Registry (PRODUCTION)
├── nbagbs-keyvault                        <-- Key Vault
├── nbagbsvmodel-env                       <-- Container Apps Environment
│   └── nba-picks-api                      <-- Container App (PRODUCTION)
│       ├── Image: nba-picks-api:v6.10
│       ├── Registry: nbagbsacr.azurecr.io
│       ├── Port: 8090
│       ├── Scaling: 1-3 replicas
│       └── Environment Variables
│           ├── THE_ODDS_API_KEY
│           ├── API_BASKETBALL_KEY
│           └── SEASONS_TO_PROCESS=2025-2026
└── (DECOMMISSIONED: nbagbsvmodel-api, nbagbsvmodelacr - to be deleted)
```

## Resource Names (NEVER CHANGES)

| Resource | Name | Status |
|----------|------|--------|
| **Resource Group** | `NBAGBSVMODEL` | Active |
| **Container Apps Environment** | `nbagbsvmodel-env` | Active |
| **Container App** | `nba-picks-api` | Active |
| **Container Registry** | `nbagbsacr` | Active |
| **Key Vault** | `nbagbs-keyvault` | Active |

## Get Current API URL (DYNAMICALLY)

The FQDN can change when Azure recreates the environment. Always get it dynamically:

```bash
# Get current API FQDN
az containerapp show -n nba-picks-api -g NBAGBSVMODEL \
  --query properties.configuration.ingress.fqdn -o tsv

# Set as environment variable
export NBA_API_URL="https://$(az containerapp show -n nba-picks-api -g NBAGBSVMODEL --query properties.configuration.ingress.fqdn -o tsv)"

# Test health
curl "$NBA_API_URL/health"
```

## Environment Variables

All scripts use these environment variables (no hardcoded values):

| Variable | Default | Description |
|----------|---------|-------------|
| `NBA_API_URL` | `http://localhost:${NBA_API_PORT}` | Full API URL |
| `NBA_API_PORT` | `8090` | Local container port |
| `TEAMS_WEBHOOK_URL` | (required) | Teams webhook URL |
| `ALLOWED_ORIGINS` | `*` | CORS origins |

## Quick Commands

```bash
# Deploy new version (recommended way)
pwsh infra/nba/deploy.ps1 -Tag v6.10

# View container app logs
az containerapp logs show -n nba-picks-api -g NBAGBSVMODEL --follow

# Update container app with new image
az containerapp update -n nba-picks-api -g NBAGBSVMODEL \
  --image nbagbsacr.azurecr.io/nba-picks-api:v6.10

# Build and push new image
docker build -t nbagbsacr.azurecr.io/nba-picks-api:v6.10 -f Dockerfile.combined .
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-picks-api:v6.10

# Run locally with custom port (avoids conflicts)
NBA_API_PORT=9000 docker compose up -d

# Check current scaling
az containerapp show -n nba-picks-api -g NBAGBSVMODEL \
  --query properties.template.scale
```

## CI/CD

GitHub Actions workflow (`.github/workflows/gbs-nba-deploy.yml`) automatically:
1. Builds Docker image on push to `main`/`master`
2. Pushes to `nbagbsacr.azurecr.io`
3. Deploys to `nba-picks-api` in `NBAGBSVMODEL`

## Model Version

- **Current Deployed:** v6.10 (9 markets: Q1 + 1H + FG)
- **Markets:** q1_spread, q1_total, q1_moneyline, 1h_spread, 1h_total, 1h_moneyline, fg_spread, fg_total, fg_moneyline
- **Dockerfile:** `Dockerfile.combined`

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check, model count, version |
| `/slate/{date}` | Get picks for a date (YYYY-MM-DD) |
| `/picks/html` | Interactive HTML page with picks |

## Important Files

| File | Purpose |
|------|---------|
| `azure/function_app/function_app.py` | Azure Function trigger + Teams webhook |
| `.github/workflows/gbs-nba-deploy.yml` | CI/CD pipeline |
| `infra/nba/main.bicep` | Infrastructure as Code |
| `infra/shared/main.bicep` | Shared resources (ACR, Environment) |
| `Dockerfile` | Container definition |
| `.env.example` | Environment variable template |

