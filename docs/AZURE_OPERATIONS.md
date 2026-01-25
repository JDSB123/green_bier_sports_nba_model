# Azure Operations

**Last Updated:** 2026-01-23
**Status:** Consolidated from AZURE_CONFIG, AZURE_CREDENTIALS_GUIDE, AZURE_CONTAINER_APP_TROUBLESHOOTING, TRACK2_DEPLOYMENT_CONFIG

---

## Resource Overview

**Resource Group:** `nba-gbsv-model-rg`

| Resource | Name | Purpose |
|----------|------|---------|
| Container App | `nba-gbsv-api` | Production API |
| Container Registry | `nbagbsacr` | Docker images |
| Key Vault | `nbagbs-keyvault` | API keys |
| Storage Account | `nbagbsvstrg` | Training data, historical |
| Log Analytics | `gbs-logs-prod` | Logging |
| App Insights | `gbs-insights-prod` | APM |
| Environment | `nba-gbsv-model-env` | Container Apps env |

---

## Deployment Pipeline

### Step 1: Commit & Push

```bash
git add .
git commit -m "feat: your changes"
git push origin main
```

### Step 2: Build Docker Image

```bash
VERSION=$(cat VERSION)
docker build \
  --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .
```

### Step 3: Push to ACR

```bash
VERSION=$(cat VERSION)
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

### Step 4: Deploy to Container App

```bash
VERSION=$(cat VERSION)
az containerapp update \
  -n nba-gbsv-api \
  -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

### Step 5: Verify

```bash
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health"
```

---

## Quick Commands

```bash
# Get API URL
export NBA_API_URL="https://$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)"

# View logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --follow

# Restart container
az containerapp restart -n nba-gbsv-api -g nba-gbsv-model-rg

# Check scaling
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.scale
```

---

## Azure Credentials (OIDC)

We use federated credentials (OIDC) for GitHub Actions. No static secrets.

### Required GitHub Secrets

| Secret | How to Find |
|--------|-------------|
| `AZURE_SUBSCRIPTION_ID` | `az account show --query id -o tsv` |
| `AZURE_TENANT_ID` | `az account show --query tenantId -o tsv` |
| `AZURE_CLIENT_ID` | App Registration → Application ID |

### Create App Registration (if needed)

1. Azure Portal → Azure Active Directory → App registrations
2. New registration: `gbs-nba-github`
3. Add federated credential:
   - Organization: `JDSB123`
   - Repository: `green_bier_sports_nba_model`
   - Entity type: Branch → `main`

---

## Troubleshooting

### Container Exits Immediately

**Check 1: Secrets not configured**
```bash
az keyvault secret list --vault-name nbagbs-keyvault --query "[].name" -o table
```

**Fix:** Add secrets to Key Vault
```bash
az keyvault secret set --vault-name nbagbs-keyvault \
  --name THE-ODDS-API-KEY --value "your_key"
```

**Check 2: Python import errors**
```bash
docker run --rm nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  python -c "from src.config import settings; print('OK')"
```

**Fix:** Rebuild image
```bash
docker build --no-cache -f Dockerfile.combined .
```

### Health Check Failing

```bash
# Check container logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg

# Verify in Azure after deploy
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl -s "https://$FQDN/health"
```

### OIDC Login Failed

1. Verify federated credential in App Registration
2. Check entity type matches (Branch: main)
3. Verify repository name is correct

---

## Infrastructure as Code

**Entry Points:**
- `infra/nba/main.bicep` - Full stack (platform + compute + optional Teams bot)
- `infra/nba/prediction.bicep` - Prediction-only (no Teams bot)

**Deploy:**
```bash
pwsh ./infra/nba/deploy.ps1 -Tag $(cat VERSION)

# What-if (preview)
pwsh ./infra/nba/deploy.ps1 -WhatIf
```

---

## Storage Containers

| Container | Purpose |
|-----------|---------|
| `models` | Model artifacts |
| `predictions` | Prediction outputs |
| `results` | Backtest results |
| `nbahistoricaldata` | Historical data (training, odds, picks) |

---

## Custom Domains

| Domain | Bound To |
|--------|----------|
| `api.greenbiersportventures.com` | nba-gbsv-api |
| `nba.greenbiersportventures.com` | nba-gbsv-api |

Managed certificates auto-renew via Azure.
