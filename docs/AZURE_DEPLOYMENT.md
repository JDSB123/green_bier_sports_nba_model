# Azure Deployment Guide

This document describes how to deploy the NBA v5.1 model to Azure.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Azure Infrastructure                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐      ┌─────────────────────────────────────┐   │
│  │  Azure Container    │      │  Azure Container Apps               │   │
│  │  Registry (ACR)     │─────▶│  nba-picks-api                      │   │
│  │  nba-v51-final:tag  │      │  ├─ /health                         │   │
│  └─────────────────────┘      │  ├─ /slate/{date}                   │   │
│                               │  └─ /slate/{date}/executive         │   │
│                               └──────────────┬──────────────────────┘   │
│                                              │                          │
│  ┌─────────────────────┐                     │                          │
│  │  Azure Function App │◀────────────────────┘                          │
│  │  nba-picks-trigger  │                                                │
│  │  ├─ /api/nba-picks  │─────────────────▶ Microsoft Teams              │
│  │  └─ /api/health     │                                                │
│  └─────────────────────┘                                                │
│                                                                         │
│  ┌─────────────────────┐      ┌─────────────────────────────────────┐   │
│  │  Log Analytics      │◀─────│  Application Insights               │   │
│  │  Workspace          │      │  (Monitoring & Telemetry)           │   │
│  └─────────────────────┘      └─────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Azure CLI** installed and logged in
2. **Docker** installed and running
3. **Azure subscription** with Contributor access
4. **API Keys**:
   - `THE_ODDS_API_KEY` - for fetching betting lines
   - `TEAMS_WEBHOOK_URL` (optional) - for posting picks to Teams

## Quick Deploy

### Option 1: PowerShell Script

```powershell
cd infra
.\deploy.ps1 -ResourceGroup "nba-picks-rg" -TheOddsApiKey "<your-key>"
```

### Option 2: Azure CLI

```bash
# Create resource group
az group create -n nba-picks-rg -l eastus

# Deploy infrastructure
az deployment group create \
  -g nba-picks-rg \
  -f infra/main.bicep \
  --parameters theOddsApiKey=<your-key>

# Build and push container
az acr login -n <acr-name>
docker build -t <acr-name>.azurecr.io/nba-v51-final:latest .
docker push <acr-name>.azurecr.io/nba-v51-final:latest

# Update container app
az containerapp update \
  -n nba-picks-api \
  -g nba-picks-rg \
  --image <acr-name>.azurecr.io/nba-v51-final:latest
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
  --scopes /subscriptions/<subscription-id>/resourceGroups/nba-picks-rg \
  --sdk-auth
```

Copy the JSON output to `AZURE_CREDENTIALS` secret.

### Get ACR Credentials

```bash
az acr credential show -n <acr-name>
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

### Container App

| Variable | Description |
|----------|-------------|
| `NBA_MODEL_VERSION` | Model version (5.1-FINAL) |
| `NBA_MARKETS` | Enabled markets |
| `THE_ODDS_API_KEY` | The Odds API key |

### Function App

| Variable | Description |
|----------|-------------|
| `NBA_API_URL` | Container App URL |
| `TEAMS_WEBHOOK_URL` | Microsoft Teams webhook |

## Monitoring

View logs in Azure Portal:
- **Container App**: Container Apps → nba-picks-api → Logs
- **Function App**: Function Apps → nba-picks-trigger → Monitor

Or via CLI:
```bash
# Container App logs
az containerapp logs show -n nba-picks-api -g nba-picks-rg

# Function App logs
az webapp log tail -n nba-picks-trigger -g nba-picks-rg
```

## Scaling

The Container App is configured with:
- **Min replicas**: 0 (scales to zero when idle)
- **Max replicas**: 3
- **Scale trigger**: 50 concurrent HTTP requests

Adjust in `infra/main.bicep` under `scale` section.

## Costs

Estimated monthly costs (East US):
- Container Apps (Consumption): ~$5-20/month (usage-based)
- Function App (Consumption): ~$0-5/month (usage-based)
- Container Registry (Basic): ~$5/month
- Log Analytics: ~$2-5/month

**Total**: ~$12-35/month depending on usage
