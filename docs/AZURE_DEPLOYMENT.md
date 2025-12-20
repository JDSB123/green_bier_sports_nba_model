# Azure Deployment Guide

**SINGLE SOURCE OF TRUTH - See AZURE_CONFIG.md**

## Actual Azure Architecture

```
greenbier-enterprise-rg                    <-- Resource Group
├── greenbieracr                           <-- Container Registry
├── greenbier-nba-env                      <-- Container Apps Environment
│   └── nba-picks-api                      <-- Container App
│       ├── Image: nba-model:v5.1
│       ├── Registry: greenbieracr.azurecr.io
│       ├── /health
│       ├── /slate/{date}
│       └── /slate/{date}/executive
└── nba-picks-trigger                      <-- Function App
    ├── /api/nba-picks
    └── /api/health
```

## Resource Names (NEVER HARDCODE FQDNs!)

| Resource | Name |
|----------|------|
| **Resource Group** | `greenbier-enterprise-rg` |
| **Container Apps Environment** | `greenbier-nba-env` |
| **Container App** | `nba-picks-api` |
| **Container Registry** | `greenbieracr` |
| **Function App** | `nba-picks-trigger` |

## Prerequisites

1. **Azure CLI** installed and logged in
2. **Docker** installed and running
3. **Azure subscription** with Contributor access
4. **API Keys** in `./secrets/` directory:
   - `THE_ODDS_API_KEY`
   - `API_BASKETBALL_KEY`

## Quick Deploy

### Option 1: PowerShell Script

```powershell
cd infra
.\deploy-enterprise.ps1 -TheOddsApiKey "<your-key>" -ApiBasketballKey "<your-key>"
```

### Option 2: Azure CLI

```bash
# Get current API FQDN (dynamically - never hardcode!)
FQDN=$(az containerapp show -n nba-picks-api -g greenbier-enterprise-rg \
  --query properties.configuration.ingress.fqdn -o tsv)
echo "API URL: https://$FQDN"

# Build and push container
az acr login -n greenbieracr
docker build -t greenbieracr.azurecr.io/nba-model:latest .
docker push greenbieracr.azurecr.io/nba-model:latest

# Update container app
az containerapp update \
  -n nba-picks-api \
  -g greenbier-enterprise-rg \
  --image greenbieracr.azurecr.io/nba-model:latest
```

## CI/CD with GitHub Actions

### Required Secrets

Add these secrets to your GitHub repository:

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | Service principal JSON for Azure login |
| `ACR_USERNAME` | Azure Container Registry admin username |
| `ACR_PASSWORD` | Azure Container Registry admin password |

### Create Service Principal

```bash
az ad sp create-for-rbac \
  --name "nba-picks-github" \
  --role contributor \
  --scopes /subscriptions/<subscription-id>/resourceGroups/greenbier-enterprise-rg \
  --sdk-auth
```

Copy the JSON output to `AZURE_CREDENTIALS` secret.

### Get ACR Credentials

```bash
az acr credential show -n greenbieracr
```

## Endpoints

### Container App API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /slate/today` | Today's predictions |
| `GET /slate/{date}` | Predictions for specific date |
| `GET /slate/{date}/executive` | Executive summary |
| `GET /slate/{date}/comprehensive` | Full analysis |

### Function App

| Endpoint | Description |
|----------|-------------|
| `GET /api/nba-picks` | Fetch picks and post to Teams |
| `GET /api/nba-picks?date=2025-12-19` | Picks for specific date |
| `GET /api/nba-picks?post=false` | Return JSON without posting |
| `GET /api/health` | Function health check |

## Environment Variables

All from environment - no hardcoded values!

| Variable | Description |
|----------|-------------|
| `NBA_API_URL` | Get dynamically from az CLI |
| `NBA_API_PORT` | Local port (default: 8090) |
| `TEAMS_WEBHOOK_URL` | Microsoft Teams webhook |
| `ALLOWED_ORIGINS` | CORS origins |

## Monitoring

View logs in Azure Portal:
- **Container App**: Container Apps → nba-picks-api → Logs
- **Function App**: Function Apps → nba-picks-trigger → Monitor

Or via CLI:
```bash
# Container App logs
az containerapp logs show -n nba-picks-api -g greenbier-enterprise-rg --follow

# Function App logs
az webapp log tail -n nba-picks-trigger -g greenbier-enterprise-rg
```

## Scaling

The Container App is configured with:
- **Min replicas**: 1 (always-on)
- **Max replicas**: 3
- **Scale trigger**: 50 concurrent HTTP requests

## Costs

Estimated monthly costs (East US):
- Container Apps (Consumption): ~$5-20/month (usage-based)
- Function App (Consumption): ~$0-5/month (usage-based)
- Container Registry (Basic): ~$5/month
- Log Analytics: ~$2-5/month

**Total**: ~$12-35/month depending on usage
