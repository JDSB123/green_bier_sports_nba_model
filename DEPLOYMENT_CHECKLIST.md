# Deployment Checklist

**Branch:** `prepare-deployment`  
**Date:** December 29, 2025  
**Version:** See `VERSION`

---

## Pre-Deployment Verification

Run the deployment preparation script:

```powershell
python scripts/prepare_deployment.py
```

This verifies:
- ✅ Version consistency across all files
- ✅ Required files present
- ✅ Docker configuration correct
- ✅ Azure Function endpoints defined
- ✅ Model files present

---

## Deployment Checklist

### 1. Code Verification ✅

- [x] All code changes committed
- [x] Version consistent (matches `VERSION`)
- [x] No uncommitted changes
- [x] Branch is `prepare-deployment` (ready for PR)

### 2. Version Consistency ✅

- [x] `VERSION` file populated
- [x] `models/production/model_pack.json` matches `VERSION`
- [x] `models/production/feature_importance.json` matches `VERSION`

### 3. Required Files ✅

- [x] `Dockerfile.combined` exists
- [x] `docker-compose.yml` exists
- [x] `requirements.txt` exists
- [x] Model files present (4 models: 1H + FG)
- [x] Azure Function code present

### 4. Docker Configuration ✅

- [x] All required ENV vars set in Dockerfile
- [x] Health check configured
- [x] Service configured in docker-compose.yml

### 5. Azure Configuration ✅

- [x] Function endpoints defined
- [x] Required functions present
- [x] Function App file exists

---

## Deployment Steps

### Step 1: Merge to Main

```powershell
# Create PR and merge prepare-deployment → main
# Or merge locally:
git checkout main
git merge prepare-deployment
git push origin main
```

### Step 2: Deploy via CI/CD (Automatic)

**Option A: Automatic (Recommended)**
- Push to `main` triggers GitHub Actions workflow
- Workflow builds Docker image and deploys automatically
- Monitor: `.github/workflows/gbs-nba-deploy.yml`

**Option B: Manual Deployment**

```powershell
# 1. Build Docker image
$VERSION = (Get-Content VERSION -Raw).Trim()
docker build --build-arg MODEL_VERSION=$VERSION -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined .

# 2. Login to ACR
az acr login -n nbagbsacr

# 3. Push image
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# 4. Deploy to Azure
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

### Step 3: Verify Deployment

```powershell
# Get FQDN
$FQDN = az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv

# Test health
curl "https://$FQDN/health" | ConvertFrom-Json

# Test predictions
curl "https://$FQDN/slate/today/executive" | ConvertFrom-Json

# Check logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --tail 50
```

---

## Post-Deployment Verification

### Health Check

```json
{
  "status": "ok",
  "version": "<VERSION>",
  "engine_loaded": true,
  "markets": 4,
  "markets_list": ["1h_spread", "1h_total", "fg_spread", "fg_total"]
}
```

### Endpoint Tests

- [ ] `/health` - Returns OK
- [ ] `/slate/today` - Returns predictions
- [ ] `/slate/today/executive` - Returns executive summary
- [ ] `/markets` - Lists all 4 markets
- [ ] `/verify` - Model integrity check passes

### Azure Function Tests

- [ ] `/api/nba-picks` - Fetches and posts picks
- [ ] `/api/health` - Health check works
- [ ] `/api/weekly-lineup/nba` - Website integration works

---

## Rollback Plan

If deployment fails:

```powershell
# Get previous image tag
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.template.containers[0].image

# Rollback to previous version
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image <previous-image-tag>
```

---

## Notes

- **API Keys:** Must be configured in Azure Container App environment variables
- **Secrets:** Managed via Azure Key Vault or Container App secrets
- **CI/CD:** Automatic deployment on push to `main` branch
- **Version:** See `VERSION` (4 markets: 1H + FG spreads/totals)

---

**Status:** ✅ Ready for deployment  
**Next:** Merge `prepare-deployment` → `main` and deploy



