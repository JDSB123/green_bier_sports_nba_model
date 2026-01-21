# Azure Container App Startup Verification Checklist

**Purpose:** Verify your deployment is correctly configured before running predictions.

**Run this BEFORE deploying to Azure:**

```bash
# 1. Ensure local code is working
python scripts/run_slate.py --date today

# 2. Verify Docker builds locally
docker build -t nba-test:latest -f Dockerfile.combined .
docker-compose up -d
curl http://localhost:8090/health | jq .

# 3. Verify version consistency
VERSION=$(cat VERSION)
echo "Local VERSION: $VERSION"
grep "version" models/production/model_pack.json | head -1
grep "git_tag" models/production/model_pack.json | head -1
# All three must match
```

---

## **Deployment Pre-Flight Checklist**

### **Git & Repo State**
- [ ] `git status` shows clean working directory (no uncommitted changes)
- [ ] `git log --oneline -1` shows recent commit
- [ ] VERSION file matches the version you want to deploy
- [ ] Recent changes are pushed: `git push origin main`

### **Version Consistency**
```bash
VERSION=$(cat VERSION)
echo "VERSION: $VERSION"
grep "\"version\"" models/production/model_pack.json | head -1
grep "\"git_tag\"" models/production/model_pack.json | head -1
```
- [ ] All three show the same version

### **Model Files**
```bash
ls -lh models/production/*.joblib
```
- [ ] 4 files exist (1h_spread, 1h_total, fg_spread, fg_total)
- [ ] Each is > 50MB (not empty)

### **Training Data**
```bash
ls -lh data/processed/training_data.csv
```
- [ ] File exists and is > 1MB

### **Docker Build (Local Test)**
```bash
$VERSION = (Get-Content VERSION).Trim()
docker build --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .
```
- [ ] Build completes with no errors
- [ ] Takes ~2-3 minutes

### **Docker Run (Local Test)**
```bash
docker-compose up -d
sleep 10
curl http://localhost:8090/health | jq .
```
- [ ] Container starts successfully
- [ ] Health endpoint returns: `"status": "ok"`
- [ ] Version in response matches VERSION file

### **Local Prediction Test**
```bash
python scripts/run_slate.py --date today
```
- [ ] Predictions generated successfully
- [ ] Predictions look reasonable (confidence 40-90%, edges ±1-5 pts)

### **Azure Secrets (Key Vault)**
```bash
az keyvault secret list --vault-name nbagbs-keyvault --query "[].name" -o table
```
- [ ] `THE-ODDS-API-KEY` exists
- [ ] `API-BASKETBALL-KEY` exists

### **Deploy to Azure**
```powershell
# Using automated script (recommended)
./scripts/deploy.ps1

# Or manual steps:
az login
$VERSION = (Get-Content VERSION).Trim()
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```
- [ ] Deploy script completes without errors
- [ ] Container App health check passes

### **Post-Deployment Verification**
```bash
$FQDN = (az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)

# Check health
curl https://$FQDN/health | jq .

# Check version matches
curl https://$FQDN/health | jq .version

# Get a test prediction
curl "https://$FQDN/slate/today/executive" | jq '.plays | length'
```
- [ ] Health endpoint returns `"status": "ok"`
- [ ] Version in response matches VERSION file
- [ ] Can fetch predictions (should return >= 0 plays)

---

## **If Deployment Fails:**

### **Troubleshooting Flowchart**

1. **Container crashes immediately (exit code 1-2)**
   - Check: Secrets configured in Key Vault
   - Check: API keys are valid
   - Check: Models exist in image
   - See: AZURE_CONTAINER_APP_TROUBLESHOOTING.md → "Scenario 1"

2. **Health check returns 500**
   - Check: `az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 30m`
   - Look for: `StartupIntegrityError`, `ModelNotFoundError`, `MissingFeaturesError`
   - See: AZURE_CONTAINER_APP_TROUBLESHOOTING.md → "Scenario 2"

3. **Predictions fail or return no picks**
   - Check: Version mismatch (`/health` version vs VERSION file)
   - Check: Feature validation errors in logs
   - Try: `curl https://$FQDN/verify` to test model integrity
   - See: AZURE_CONTAINER_APP_TROUBLESHOOTING.md → "Scenario 3"

4. **Deploy script hangs or times out**
   - Check: Azure CLI authentication (`az account show`)
   - Check: Docker image pushed successfully (`az acr repository list -n nbagbsacr`)
   - See: AZURE_CONTAINER_APP_TROUBLESHOOTING.md → "Scenario 4"

---

## **Common Issues & Quick Fixes**

### **Issue: "Container exits immediately"**
```bash
# Check what's happening
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 5m

# If you see "No such file or directory" for models:
# → Models weren't copied into Docker image
# → Make sure models are committed to git before building

# If you see import errors:
# → Rebuild Docker image from scratch
$VERSION = (Get-Content VERSION).Trim()
docker build --no-cache --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .
```

### **Issue: "Health check returns 503"**
```bash
# The engine failed to load - check what's wrong
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 30m | grep -i "error\|fail"

# If models are missing:
python scripts/train_models.py
git add models/production/
git commit -m "Add trained models for $(cat VERSION)"
git push origin main

# Rebuild and redeploy
# (Then use deploy.ps1 script)
```

### **Issue: "Predictions have very low accuracy (worse than local)"**
```bash
# Check version mismatch
curl https://$FQDN/health | jq .version
cat VERSION
# They must match!

# If they don't match:
$VERSION = (Get-Content VERSION).Trim()
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Wait for deployment to finish
sleep 30
curl https://$FQDN/health | jq .version
```

### **Issue: "API returns empty picks despite games scheduled"**
```bash
# Check if splits provider is down
curl https://$FQDN/health | jq .betting_splits_sources

# If all return errors, fallback mode works:
python scripts/run_slate.py --date today --use-splits false

# For Azure, create a manual test:
curl "https://$FQDN/slate/today?use_splits=false" | jq '.predictions | length'
```

---

## **After Successful Deployment**

Monitor these metrics:

```bash
# View live logs (press Ctrl+C to stop)
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --follow

# Get resource usage
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].resources

# Track recent errors
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 1h \
  | grep -i "error\|warn\|fail"

# Test a prediction
curl "https://$FQDN/slate/today/executive" | jq '.plays | length'
```

---

## **References**

- [AZURE_CONTAINER_APP_TROUBLESHOOTING.md](AZURE_CONTAINER_APP_TROUBLESHOOTING.md) - Deep diagnostics
- [DOCKER_SECRETS.md](DOCKER_SECRETS.md) - Secret management
- [.github/copilot-instructions.md](../.github/copilot-instructions.md) - Deployment pipeline
- [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Full deployment guide
