# Azure Configuration - SINGLE SOURCE OF TRUTH

**Last Updated:** 2025-12-19

## Actual Azure Architecture

```
greenbier-enterprise-rg                    <-- Resource Group
├── greenbieracr                           <-- Container Registry
├── greenbier-nba-env                      <-- Container Apps Environment
│   └── nba-picks-api                      <-- Container App
│       ├── Image: nba-model:v5.1
│       ├── Registry: greenbieracr.azurecr.io
│       ├── Scaling: 1-3 replicas
│       └── Environment Variables
│           ├── NBA_MODEL_VERSION
│           └── NBA_MARKETS
└── nba-picks-trigger                      <-- Function App (optional)
```

## Resource Names (NEVER CHANGES)

| Resource | Name |
|----------|------|
| **Resource Group** | `greenbier-enterprise-rg` |
| **Container Apps Environment** | `greenbier-nba-env` |
| **Container App** | `nba-picks-api` |
| **Container Registry** | `greenbieracr` |
| **Function App** | `nba-picks-trigger` |

## Get Current API URL (DYNAMICALLY)

The FQDN can change when Azure recreates the environment. Always get it dynamically:

```bash
# Get current API FQDN
az containerapp show -n nba-picks-api -g greenbier-enterprise-rg \
  --query properties.configuration.ingress.fqdn -o tsv

# Set as environment variable
export NBA_API_URL="https://$(az containerapp show -n nba-picks-api -g greenbier-enterprise-rg --query properties.configuration.ingress.fqdn -o tsv)"

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
# View container app logs
az containerapp logs show -n nba-picks-api -g greenbier-enterprise-rg --follow

# Update container app with new image
az containerapp update -n nba-picks-api -g greenbier-enterprise-rg \
  --image greenbieracr.azurecr.io/nba-model:latest

# Build and push new image
docker build -t greenbieracr.azurecr.io/nba-model:latest .
az acr login -n greenbieracr
docker push greenbieracr.azurecr.io/nba-model:latest

# Run locally with custom port (avoids conflicts)
NBA_API_PORT=9000 docker compose up -d

# Check current scaling
az containerapp show -n nba-picks-api -g greenbier-enterprise-rg \
  --query properties.template.scale
```

## CI/CD

GitHub Actions workflow (`.github/workflows/gbs-nba-deploy.yml`) automatically:
1. Builds Docker image on push to `main`/`master`
2. Pushes to `greenbieracr.azurecr.io`
3. Deploys to `nba-picks-api` in `greenbier-enterprise-rg`

## Model Version

- **Current Deployed:** v5.1 (6 markets: FG + 1H)
- **Target:** v6.0 (9 markets: Q1 + 1H + FG)

## Important Files

| File | Purpose |
|------|---------|
| `azure/function_app/function_app.py` | Azure Function trigger + Teams webhook |
| `.github/workflows/gbs-nba-deploy.yml` | CI/CD pipeline |
| `infra/nba/main.bicep` | Infrastructure as Code |
| `infra/shared/main.bicep` | Shared resources (ACR, Environment) |
| `Dockerfile` | Container definition |
| `.env.example` | Environment variable template |
