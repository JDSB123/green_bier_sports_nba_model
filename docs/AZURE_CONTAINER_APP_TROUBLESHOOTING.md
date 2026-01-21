# Azure Container App Deployment Troubleshooting

**Last Updated:** 2026-01-21  
**Version:** NBA_v33.1.3

## Common Failure Scenarios & Fixes

---

## **Scenario 1: Container Exits Immediately (Exit Code 1-2)**

### Symptoms
- Container restarts repeatedly
- Health check returns `503` or timeout
- `az containerapp logs` shows nothing or minimal output

### Root Causes & Fixes

#### **A. Secrets Not Configured in Azure Key Vault**

**Problem:** Container app can't load API keys from Key Vault.

**Check:**
```powershell
# Verify secrets exist in Key Vault
az keyvault secret list --vault-name nbagbs-keyvault --query "[].name" -o table

# Expected secrets:
# - THE-ODDS-API-KEY
# - API-BASKETBALL-KEY
```

**Fix:** Add secrets to Key Vault
```powershell
# If missing, fetch from local .env or get from external provider
THE_ODDS_KEY=$(cat secrets/THE_ODDS_API_KEY)
az keyvault secret set --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --value "$THE_ODDS_KEY"

API_BASKETBALL_KEY=$(cat secrets/API_BASKETBALL_KEY)
az keyvault secret set --vault-name nbagbs-keyvault --name API-BASKETBALL-KEY --value "$API_BASKETBALL_KEY"
```

**Verify in Container App:**
```powershell
# Check environment variables are wired in Container App
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].env -o json

# Should show:
# {
#   "name": "THE_ODDS_API_KEY",
#   "secretRef": "the-odds-api-key"  
# }
```

---

#### **B. Python Import Errors (Missing Dependencies)**

**Problem:** Dependencies not installed in Docker image.

**Check:**
```bash
# SSH into container (if possible) or check logs
docker run --rm nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.3 \
  python -c "from src.config import settings; print('OK')"
```

**Fix:** Rebuild Docker image
```powershell
# From repo root:
$VERSION = (Get-Content VERSION).Trim()
docker build --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .

# If rebuild fails, check requirements.txt is up to date
pip freeze > requirements.txt
```

---

#### **C. Training Data File Missing**

**Problem:** `data/processed/training_data.csv` not available in container.

**Check:**
```powershell
# Verify file exists locally BEFORE deployment
Test-Path "data/processed/training_data.csv"

# Check it's being copied in Dockerfile
Select-String "COPY data/processed" Dockerfile.combined
```

**Fix:** Ensure data is tracked in git or rebuild training data
```bash
# Option 1: Add training data to git (if < 100MB)
git add data/processed/training_data.csv
git commit -m "Add production training data"
git push origin main

# Option 2: Rebuild training data before deployment
python scripts/build_training_data_complete.py

# Option 3: Mount as Azure Blob Storage volume (advanced)
```

---

## **Scenario 2: Health Check Fails (Returns 500)**

### Symptoms
- Container runs but `GET /health` returns `HTTP 500`
- Deploy script fails at health verification
- `az containerapp logs` shows `StartupIntegrityError` or `ModelNotFoundError`

### Root Causes & Fixes

#### **A. Model Files Missing or Corrupt**

**Problem:** Models weren't trained or weren't copied into Docker image.

**Check:**
```powershell
# Expected models in Docker image:
docker run --rm nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.3 \
  find /app/models/production -type f -name "*.joblib"

# Should show 4 files:
# /app/models/production/1h_spread_model.joblib
# /app/models/production/1h_total_model.joblib
# /app/models/production/fg_spread_model.joblib
# /app/models/production/fg_total_model.joblib
```

**Fix:** Train models and rebuild Docker image
```bash
# Train models locally
python scripts/train_models.py

# Verify they exist
ls -lh models/production/*.joblib

# Commit to git
git add models/production/
git commit -m "Add trained models for v33.1.3"
git push origin main

# Then rebuild container
$VERSION = (Get-Content VERSION).Trim()
docker build --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

---

#### **B. Startup Integrity Check Fails**

**Problem:** `src/utils/startup_checks.py` detects configuration errors.

**Check:**
```bash
# Run startup checks locally
python -c "from src.utils.startup_checks import run_startup_integrity_checks; run_startup_integrity_checks()"

# Or enable verbose logging:
export LOGLEVEL=DEBUG
python scripts/predict.py  # This triggers app startup
```

**Common failures:**
- Feature configuration mismatch between trained model and config
- Missing API keys in environment
- Unsupported Python version

**Fix:** 
```bash
# Check feature configuration
python -c "from src.modeling.unified_features import get_all_features; print(len(get_all_features()))"

# Verify config matches VERSION
grep "NBA_v" models/production/model_pack.json | head -5
```

---

#### **C. API Key Status Check Fails**

**Problem:** API keys are configured but invalid or expired.

**Check:**
```powershell
# List current secrets in Key Vault
az keyvault secret show --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --query value -o tsv | Measure-Object -Character

# Test API key validity by trying a simple request
curl -s "https://api.the-odds-api.com/v4/sports/basketball_nba/participants?apiKey=<KEY>" | head -c 200
```

**Fix:** Rotate keys in Key Vault
```powershell
# Update the secret
az keyvault secret set --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --value "<NEW_KEY>"

# The Container App will pick up the new value on next deployment/restart
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.3
```

---

## **Scenario 3: Predictions Fail After Deployment (Returns Data but Accuracy is Wrong)**

### Symptoms
- API returns predictions
- But predictions are much worse than local predictions
- Feature values are different or missing

### Root Causes & Fixes

#### **A. Version Mismatch (Old Code, New Models)**

**Problem:** Deployed code doesn't match the models that were trained.

**Example:**
- Local repo: `v33.1.3` with new feature engineering
- Container: still running `v33.1.1` with old features
- Result: Feature names don't match → silent zero-filling → bad predictions

**Check:**
```powershell
# Check deployed version
$FQDN = (az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl https://$FQDN/health | jq .version

# Compare to local VERSION
Get-Content VERSION

# They MUST match
```

**Fix:** Redeploy with correct version
```powershell
# Make sure local workspace is up-to-date
git pull origin main
$VERSION = (Get-Content VERSION).Trim()

# Rebuild and push (this forces a rebuild)
docker build --build-arg MODEL_VERSION=$VERSION --no-cache \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Update Container App
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

Or use the automated deploy script:
```powershell
./scripts/deploy.ps1
```

---

#### **B. Feature Engineering Differences (Splits Not Available)**

**Problem:** Betting splits endpoint is down or not wired in Azure.

**Check:**
```powershell
# Get health status (includes splits validation)
$FQDN = (az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl https://$FQDN/health | jq .betting_splits_sources

# Should show: { "action_network": "ok", "draftkings": "ok", ... }
# If errors, check environment variables are set:
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].env -o json | grep -i splits
```

**Fix:** If splits providers are down temporarily, use fallback mode
```bash
# Local testing without splits
python scripts/run_slate.py --use-splits false

# For Azure deployment, contact providers or disable market:
# Set FILTER_*_CONFIDENCE to very high (0.99) to block predictions without splits
```

---

#### **C. Data Ingestion Errors (Silent Failures)**

**Problem:** API calls to The Odds or API-Basketball fail, but container doesn't report it.

**Check:**
```powershell
# Check detailed app logs for ingestion errors
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 30m | Select-String -Pattern "error|fail|invalid" -Context 2

# Test API endpoints manually
curl -s "https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=$ENV:THE_ODDS_API_KEY&regions=us&markets=spread" | jq '.response | length'
```

**Fix:** 
- Check API rate limits haven't been exceeded
- Verify API keys are active in The Odds and API-Basketball dashboards
- Add detailed logging to `src/ingestion/` modules:

```python
# In src/ingestion/the_odds.py
logger.error(f"[CRITICAL] The Odds API failed: {response.status_code} - {response.text[:200]}")
```

---

## **Scenario 4: Deployment Script Hangs or Fails**

### Symptoms
- `./scripts/deploy.ps1` hangs at "Updating Azure Container App"
- Returns `az` command not found errors
- SSH into container times out

### Root Causes & Fixes

#### **A. Azure CLI Not Authenticated**

**Problem:** `az` commands fail due to missing credentials.

**Check:**
```powershell
az account show
# If this fails or shows wrong account, re-authenticate
```

**Fix:**
```powershell
# Login to Azure
az login

# Set the correct subscription
az account set --subscription "<SUBSCRIPTION_ID>"

# Verify
az account show --query "{name:name, id:id}"
```

---

#### **B. Container Registry Authentication Failed**

**Problem:** Docker can't push to ACR.

**Check:**
```powershell
# Verify ACR credentials
az acr login -n nbagbsacr --expose-token
```

**Fix:**
```powershell
# Re-authenticate to ACR
az acr login -n nbagbsacr

# If using Docker Desktop, clear credentials and re-login:
docker logout nbagbsacr.azurecr.io
az acr login -n nbagbsacr
```

---

#### **C. Container App Update Times Out**

**Problem:** `az containerapp update` hangs or times out.

**Check:**
```powershell
# Check if Container App exists and is accessible
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query "properties.provisioningState"

# Should return "Succeeded"
```

**Fix:**
```powershell
# Try with explicit timeout
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.3 \
  --no-wait  # Don't wait for deployment to complete

# Then monitor separately
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query "properties.provisioningState" -o table
```

---

## **Deployment Checklist (Before Pushing to Azure)**

```bash
# 1. Version consistency
cat VERSION
grep "version" models/production/model_pack.json | head -1

# 2. All models trained
ls -lh models/production/*.joblib | wc -l  # Should be 4

# 3. Training data exists
ls -lh data/processed/training_data.csv

# 4. Git is clean and pushed
git status
git log --oneline -1

# 5. Local Docker build succeeds
$VERSION = (Get-Content VERSION).Trim()
docker build -t test:latest -f Dockerfile.combined .

# 6. Local Docker test succeeds
docker-compose up -d
curl http://localhost:8090/health | jq .

# 7. API keys configured in Key Vault
az keyvault secret list --vault-name nbagbs-keyvault --query "[?contains(name, 'KEY')].name"

# 8. Run tests
python -m pytest tests -v -k "not integration"

# 9. Ready to deploy
./scripts/deploy.ps1 --dry-run  # Preview what will happen
./scripts/deploy.ps1             # Execute deployment
```

---

## **Emergency Rollback**

If a deployment causes widespread issues:

```powershell
# Redeploy previous working version
$PREVIOUS_VERSION = "NBA_v33.1.1"  # Known working version

az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image "nbagbsacr.azurecr.io/nba-gbsv-api:$PREVIOUS_VERSION"

# Verify
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 5m
```

---

## **Monitoring & Observability**

### View Live Logs
```powershell
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --follow
```

### Check Resource Usage
```powershell
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query "properties.template.containers[0].resources"
```

### View Recent Revisions
```powershell
az containerapp revision list -n nba-gbsv-api -g nba-gbsv-model-rg -o table
```

### Enable Application Insights (Advanced)
```powershell
# Create or link Application Insights
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --instrumentation-key "<INSTRUMENTATION_KEY>"
```

---

## **See Also**

- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Pre-deployment validation
- [DOCKER_TROUBLESHOOTING.md](DOCKER_TROUBLESHOOTING.md) - Local Docker issues
- [DOCKER_SECRETS.md](DOCKER_SECRETS.md) - Secret management
- [.github/copilot-instructions.md](../.github/copilot-instructions.md) - Deployment pipeline

