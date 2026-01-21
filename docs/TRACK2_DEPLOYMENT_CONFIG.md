# Track 2: Deployment & Azure Configuration - Single Source of Truth

**Purpose:** Ensure Version → Image Tag → Deployment steps are unambiguous and synchronized.

**Last Updated:** 2026-01-21  
**Status:** ✅ Ready for Contributors

---

## Version Management

### Single Source: `VERSION` File

```bash
cat VERSION
# Output: NBA_v33.1.4
```

**Usage:**
- Baked into Docker images as `MODEL_VERSION` build arg
- Used in ALL deployment commands
- Never hardcoded; always read from `VERSION` file

### Image Naming Convention

```
nbagbsacr.azurecr.io/nba-gbsv-api:{VERSION}

Example:
  nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.4
```

**Versioning Rule:**
- Semantic: `NBA_v{Major}.{Minor}.{Patch}`
- Updated by: Build process (push to main triggers Docker build)
- Never hardcode version in code; read from `VERSION` file

---

## Deployment Pipeline (Local to Azure)

### STEP 1: Commit & Push to GitHub Main

```bash
git add .
git commit -m "feat(models): update picks logic"
git push origin main
```

✅ This triggers the SINGLE SOURCE OF TRUTH (`main` branch)

---

### STEP 2: Build Docker Image

Build the combined image with the current VERSION:

```bash
VERSION=$(cat VERSION)
docker build \
  --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined \
  .
```

**Note:** `Dockerfile.combined` is the production image (4 markets: 1H + FG).

---

### STEP 3: Push to Azure Container Registry (ACR)

```bash
VERSION=$(cat VERSION)

# Login to ACR (one-time per session)
az acr login -n nbagbsacr

# Push image
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

✅ Image is now available at `nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.4`

---

### STEP 4: Deploy to Azure Container App

```bash
VERSION=$(cat VERSION)

az containerapp update \
  -n nba-gbsv-api \
  -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

✅ Container App pulls the image and auto-restarts with the new version

---

## Verification

### Health Check

```bash
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv)

curl "https://$FQDN/health"
```

**Expected Response:**
```json
{
  "status": "ok",
  "version": "NBA_v33.1.4",
  "engine_loaded": true,
  "markets": 4,
  "markets_list": ["1h_spread", "1h_total", "fg_spread", "fg_total"]
}
```

### Live Endpoints

```bash
# Slate (today's picks)
curl "https://$FQDN/slate/today"

# Executive summary
curl "https://$FQDN/slate/today/executive"

# Market info
curl "https://$FQDN/markets"
```

---

## Rollback

If deployment fails, revert to previous image:

```bash
# Find previous image (check Container App revision history)
az containerapp revision list -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query '[0:5].[name,properties.template.containers[0].image]' -o table

# Rollback to specific image
az containerapp update \
  -n nba-gbsv-api \
  -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.3
```

---

## Key Vault & Secrets

**Required Secrets (in `nbagbs-keyvault`):**

| Secret | Description |
|--------|-------------|
| `THE-ODDS-API-KEY` | The Odds API key |
| `API-BASKETBALL-KEY` | API-Basketball key |

**Container App retrieves from Key Vault:**

```bash
# View secret (if you have access)
az keyvault secret show \
  --vault-name nbagbs-keyvault \
  --name THE-ODDS-API-KEY

# Set new secret
az keyvault secret set \
  --vault-name nbagbs-keyvault \
  --name THE-ODDS-API-KEY \
  --value "new_key_here"
```

---

## Files (Source of Truth)

| File | Purpose | Who Edits |
|------|---------|-----------|
| `VERSION` | Current semantic version | Build/Release engineer |
| `Dockerfile.combined` | Production image definition | Code owner |
| `docker-compose.yml` | Local dev stack | Dev team |
| `.env.example` | Config template | Dev team |
| `infra/nba/main.bicep` | Azure infrastructure (IaC) | DevOps |
| `docs/AZURE_CONFIG.md` | Azure resource names/FQDN | DevOps |
| `.github/copilot-instructions.md` | Deployment runbook | Maintainer |

---

## CI/CD Automation (Future)

GitHub Actions workflow (`.github/workflows/gbs-nba-deploy.yml`):
- Trigger: push to `main`
- Steps:
  1. Read `VERSION`
  2. Build Docker image
  3. Push to ACR
  4. Update Container App
  5. Verify health

**Current Status:** Manual deployment via CLI (steps 1-4 above)

---

## Quick Reference: Full Deploy Command

```bash
# All-in-one (copy-paste)
VERSION=$(cat VERSION) && \
  docker build --build-arg MODEL_VERSION=$VERSION -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined . && \
  az acr login -n nbagbsacr && \
  docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION && \
  az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION && \
  FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv) && \
  curl "https://$FQDN/health"
```

---

## Troubleshooting

### Image not found in ACR

```bash
# Verify image exists
az acr repository list -n nbagbsacr | grep nba-gbsv-api

# List all tags for the repository
az acr repository show-tags -n nbagbsacr -r nba-gbsv-api
```

### Container App not updating

```bash
# Check revision history (latest first)
az containerapp revision list -n nba-gbsv-api -g nba-gbsv-model-rg

# Check event logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --tail 50

# Force update with no-change update trick
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --force-latest-revision true
```

### Health check fails

```bash
# View container logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --follow

# Check environment variables
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].env -o table

# Check model files in running container
az containerapp exec -n nba-gbsv-api -g nba-gbsv-model-rg \
  --command "ls -la /app/data/processed/models/"
```

---

## Summary: The Three Rules

1. **VERSION is source of truth**: Always read from `VERSION` file.
2. **Image tags match VERSION**: Never hardcode tag; derive from `VERSION`.
3. **Deploy pipeline is linear**: main → build → ACR → Container App → health check.

