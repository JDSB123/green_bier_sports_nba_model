# Azure Configuration - SINGLE SOURCE OF TRUTH

**Last Updated:** 2025-12-19

## Resource Locations (NEVER CHANGES)

| Resource | Name | Resource Group |
|----------|------|----------------|
| **Container App** | `nba-picks-api` | `greenbier-enterprise-rg` |
| **Container Registry** | `greenbieracr` | `greenbier-enterprise-rg` |
| **Function App** | `nba-picks-trigger` | `greenbier-enterprise-rg` |

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
| `NBA_API_URL` | `http://localhost:8090` | Full API URL |
| `NBA_API_PORT` | `8090` | Local container port |
| `TEAMS_WEBHOOK_URL` | (required) | Teams webhook URL |
| `ALLOWED_ORIGINS` | `*` | CORS origins |

## Quick Commands

```bash
# View container app logs
az containerapp logs show -n nba-picks-api -g greenbier-enterprise-rg --follow

# Update container app
az containerapp update -n nba-picks-api -g greenbier-enterprise-rg \
  --image greenbieracr.azurecr.io/nba-model:latest

# Build and push new image
docker build -t greenbieracr.azurecr.io/nba-model:latest .
az acr login -n greenbieracr
docker push greenbieracr.azurecr.io/nba-model:latest

# Run locally with custom port
NBA_API_PORT=9000 docker compose up -d
```

## CI/CD

GitHub Actions workflow (`.github/workflows/gbs-nba-deploy.yml`) automatically:
1. Builds Docker image on push to `main`/`master`
2. Pushes to `greenbieracr.azurecr.io`
3. Deploys to `nba-picks-api` in `greenbier-enterprise-rg`

## Model Version

- **Current:** 5.1-FINAL (6 markets: FG + 1H)
- **Target:** 6.0 (9 markets: Q1 + 1H + FG)

## Important Files

| File | Purpose |
|------|---------|
| `azure/function_app/function_app.py` | Azure Function trigger + Teams webhook |
| `.github/workflows/gbs-nba-deploy.yml` | CI/CD pipeline |
| `infra/nba/main.bicep` | Infrastructure as Code |
| `Dockerfile` | Container definition |
| `.env.example` | Environment variable template |
