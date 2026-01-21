# NBA Container App Deployment Status - Root Cause Analysis

**Date:** January 21, 2026
**Version:** NBA_v33.1.3
**Issue:** Predictions work locally but fail in Azure Container App

---

## **Root Causes Identified**

Based on analysis of your codebase and deployment pipeline, here are the **5 critical issues** causing failures:

### **1. Version Drift Detection Missing** ⚠️ CRITICAL
**Problem:** After a `git pull`, you can have code changes but the old Docker image still running in Azure, causing feature mismatches.

**Impact:**
- Features changed locally but container predicts with old model
- Confidence/edge calculations fail silently
- Predictions become inaccurate

**Status:** ✅ FIXED in recent commits
**What you need to do:** Use the updated `run_slate.py` that includes `ensure_api_version_matches_local()` function which auto-rebuilds the container if versions drift.

---

### **2. API Secrets Not Wired in Azure** ⚠️ CRITICAL
**Problem:** Container App environment variables aren't configured to pull `THE_ODDS_API_KEY` and `API_BASKETBALL_KEY` from Key Vault.

**Impact:**
- Data ingestion fails silently
- Container thinks keys are missing
- Returns empty predictions

**How to Fix:**
```powershell
# List current secrets in Key Vault
az keyvault secret list --vault-name nbagbs-keyvault --query "[].name" -o table

# If they don't exist, add them:
az keyvault secret set --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --value "<YOUR_KEY>"

# Verify Container App has environment variable bindings:
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].env -o json
```

**Expected output:**
```json
[
  {
    "name": "THE_ODDS_API_KEY",
    "secretRef": "the-odds-api-key"
  },
  {
    "name": "API_BASKETBALL_KEY",
    "secretRef": "api-basketball-key"
  }
]
```

---

### **3. FQDN Hardcoded in Deploy Script** ⚠️ MEDIUM
**Problem:** `deploy.ps1` had a hardcoded FQDN that fails when deployed to new regions or accounts.

**Impact:**
- Health check fails at end of deployment
- Deployment appears to fail even though it succeeds

**Status:** ✅ FIXED
**Fix Applied:** FQDN now dynamically resolved using `az containerapp show`

---

### **4. Health Check Status Mismatch** ⚠️ MEDIUM
**Problem:** App returns `"status": "ok"` but deploy script checked for `"status": "healthy"`.

**Impact:**
- Health verification fails even though app is running
- Deployment verification fails

**Status:** ✅ FIXED
**Fix Applied:** deploy.ps1 now accepts both `"ok"` and `"healthy"`

---

### **5. Feature Validation Too Strict in Strict Mode** ⚠️ MEDIUM
**Problem:** If ANY feature is missing during prediction, the entire request fails with 500 error instead of using defaults.

**Impact:**
- Predictions crash if external API adds new fields
- Predictions crash if external API removes fields
- No graceful degradation

**How to Debug:**
```bash
# Check what's happening:
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 30m \
  | grep -i "missing\|feature\|error"

# Test with verbose logging:
export PREDICTION_FEATURE_MODE=warn
python scripts/run_slate.py --date today
```

---

## **What's Already Fixed** ✅

Recent commits in your repo have already applied several critical fixes:

1. ✅ **Version bump to v33.1.3** - Latest models and features
2. ✅ **Critical blacklist fix** - Removed predicted_margin/total from leakage blacklist
3. ✅ **Deploy script FQDN resolution** - Now dynamically resolves Container App FQDN
4. ✅ **Health check status compatibility** - Accepts both "ok" and "healthy"
5. ✅ **Splitting fallback** - Handles when Action Network splits provider is down
6. ✅ **Version mismatch detection** - run_slate.py now detects and rebuilds on version drift

---

## **What You Need to Do NOW**

### **Immediate Actions (15 minutes)**

1. **Verify Secrets in Key Vault:**
```powershell
az keyvault secret list --vault-name nbagbs-keyvault --query "[].name" -o table

# Must show: THE-ODDS-API-KEY and API-BASKETBALL-KEY
```

2. **Verify Container App Environment Variables:**
```powershell
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.template.containers[0].env -o json
```

3. **Check Current Logs for Errors:**
```powershell
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 1h \
  | Select-String "ERROR\|CRITICAL" -Context 3
```

---

### **If Predictions Still Fail:**

**Option A: Quick Redeploy**
```powershell
# Make sure local code is up to date
git pull origin main

# Use automated deploy script
./scripts/deploy.ps1
```

**Option B: Deep Diagnosis**
```bash
# 1. Run local test first
python scripts/run_slate.py --date today

# 2. Check Docker locally
docker-compose up -d
curl http://localhost:8090/health | jq .

# 3. Check what's in Azure logs
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 30m

# 4. Test Azure health endpoint
$FQDN = (az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl https://$FQDN/health | jq .

# 5. Test actual predictions
curl "https://$FQDN/slate/today" | jq '.predictions | length'
```

---

## **Documentation Created**

I've created comprehensive troubleshooting guides:

1. **[AZURE_CONTAINER_APP_TROUBLESHOOTING.md](docs/AZURE_CONTAINER_APP_TROUBLESHOOTING.md)**
   - 4 detailed failure scenarios with root causes
   - Step-by-step fixes for each scenario
   - Includes deployment checklist and rollback procedures

2. **[AZURE_STARTUP_VERIFICATION.md](AZURE_STARTUP_VERIFICATION.md)**
   - Pre-deployment checklist
   - Post-deployment verification steps
   - Quick fixes for common issues

---

## **Key Takeaways**

| Issue | Root Cause | Status | Fix |
|-------|-----------|--------|-----|
| Version mismatch | Old code, new models | ✅ Detected | `run_slate.py` auto-rebuilds |
| Secrets missing | Key Vault not wired | ⚠️ Verify | Check env variable bindings |
| Health check fails | Hardcoded FQDN | ✅ Fixed | Dynamic resolution in place |
| Features missing | Strict validation | ⚠️ Monitor | Check logs for feature errors |
| Empty predictions | API data fetch fails | ⚠️ Verify | Check splits provider status |

---

## **Next Steps**

1. ✅ **Run verification checklist:** [AZURE_STARTUP_VERIFICATION.md](AZURE_STARTUP_VERIFICATION.md)
2. ✅ **Test locally first:** `python scripts/run_slate.py --date today`
3. ✅ **Verify Azure secrets:** `az keyvault secret list --vault-name nbagbs-keyvault`
4. ✅ **Redeploy:** `./scripts/deploy.ps1`
5. ✅ **Test Azure endpoint:** `curl https://$FQDN/health | jq .`

---

## **Emergency Rollback**

If the latest version causes issues:

```powershell
# Redeploy previous known-working version
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.1.1

# Verify
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg --since 5m
```

---

## **Contact & Support**

- **Repository:** https://github.com/JDSB123/green_bier_sports_nba_model
- **Main branch:** `main`
- **Current version:** NBA_v33.1.3
- **Resource group:** `nba-gbsv-model-rg`
- **Container app:** `nba-gbsv-api`
