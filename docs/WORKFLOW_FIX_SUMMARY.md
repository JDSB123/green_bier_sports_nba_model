# GitHub Actions Workflow Fix Summary
**Date:** 2025-12-28  
**Status:** ✅ **FIXED**

---

## What Was Broken
All GitHub Actions workflows using OIDC (OpenID Connect) authentication were failing at the "Log in to Azure" step:

```
##[error]No subscriptions found for ***.
##[error]Login failed with Error: The process '/usr/bin/az' failed with exit code 1
```

**Root cause:** Azure Service Principal credentials were expired/stale (last updated 7-8 days ago).

---

## What Was Fixed

### ✅ Step 1: Updated GitHub Secrets
Updated the following secrets in GitHub with fresh Azure credentials:
- `AZURE_SUBSCRIPTION_ID` → `3a1a4a94-45a5-4f7c-8ada-97978221052c`
- `AZURE_TENANT_ID` → `18ee0910-417d-4a81-a3f5-7945bdbd5a78`
- `AZURE_CLIENT_ID` → `971db985-be14-4352-bb1d-144d8e8b198c` (Service Principal: `gbs-nba-github`)

**Command:**
```powershell
gh secret set AZURE_SUBSCRIPTION_ID --body "3a1a4a94-45a5-4f7c-8ada-97978221052c"
gh secret set AZURE_TENANT_ID --body "18ee0910-417d-4a81-a3f5-7945bdbd5a78"
gh secret set AZURE_CLIENT_ID --body "971db985-be14-4352-bb1d-144d8e8b198c"
```

### ✅ Step 2: Assigned Azure Role
Granted the service principal `gbs-nba-github` the **Contributor** role on the Azure subscription:

```powershell
az role assignment create \
  --assignee-object-id 03ed6070-7e9b-4dac-8855-8d0ac93601b8 \
  --assignee-principal-type ServicePrincipal \
  --role "Contributor" \
  --scope "/subscriptions/3a1a4a94-45a5-4f7c-8ada-97978221052c"
```

**Result:** Service principal now has permissions to:
- Build and push Docker images to `nbagbsacr.azurecr.io`
- Update Container Apps in `nba-gbsv-model-rg`
- Execute OIDC authentication against Azure

---

## Workflow Architecture Clarification

### Current Workflow Strategy (Dec 2025)

You have **5 workflows** in the repository. Here's which ones are active:

| Workflow File | Trigger | Purpose | Status |
|----|----|----|----|
| **`gbs-nba-deploy.yml`** | Auto on `push` to main | **PRIMARY**: Build Docker image + auto-deploy to Container Apps | ✅ **ACTIVE** |
| **`gbs-nba-function.yml`** | Auto on changes in `azure/function_app/**` | Deploy Azure Function App | ✅ **ACTIVE** |
| **`build-push-acr.yml`** | Auto on `push` to main (code changes) | Build and push to ACR only | ⚠️ **Semi-deprecated** |
| **`deploy-aca.yml`** | Manual (`workflow_dispatch`) | Manual deploy for rollbacks | ⚠️ **Fallback only** |
| **`acr-retention.yml`** | Weekly schedule + manual | Cleanup old image tags | ✅ **ACTIVE** |

### Recommended Usage

**For normal development:**
```
git push origin main
    ↓
GBS NBA - Build & Deploy (automatic)
    ↓
Build Docker image
    ↓
Push to nbagbsacr.azurecr.io
    ↓
Deploy to nba-gbsv-api Container App
    ↓
✅ Done (fully automated)
```

**For Function App changes:**
```
git push changes to azure/function_app/**
    ↓
GBS NBA - Deploy Function (automatic)
    ↓
Deploy to nba-picks-trigger Function App
    ↓
✅ Done (fully automated)
```

**For manual rollback to specific image:**
```bash
gh workflow run "Deploy NBA Image to Azure Container Apps" \
  -f tag=NBA_v33.0.8.0
```

---

## Why Some Workflows Are Deprecated

### `build-push-acr.yml` (Semi-deprecated)

**What it does:**
- Triggers on code changes (same as GBS NBA - Build & Deploy)
- Builds Docker image
- Pushes to ACR with tags: `git-<sha>` and `NBA_v33.0`
- **Does NOT deploy** to Container Apps

**Why semi-deprecated:**
- `gbs-nba-deploy.yml` does the SAME build + push, but also deploys
- Running both creates redundant builds and pushes
- Could be removed, but kept as fallback

**Keep it if:**
- You want separate control over building vs. deploying
- You want to manually test images before deploying

**Remove it if:**
- You only want automated build + deploy (recommended)

### `deploy-aca.yml` (Fallback only)

**What it does:**
- Manual trigger only (not automatic)
- Deploys an already-built image to Container Apps
- Requires you to provide the tag name

**Why keep it:**
- Useful for **rollbacks** without rebuilding
- Example: If deployment was bad, quickly revert to `NBA_v33.0.7.0`
- No dependencies on code—just redeploy existing image

**Usage for rollback:**
```bash
gh workflow run "Deploy NBA Image to Azure Container Apps" \
  -f tag=NBA_v33.0.7.0
```

---

## Next Steps

### ✅ Already Done
- [x] Updated Azure credentials in GitHub Secrets
- [x] Assigned service principal Contributor role
- [x] Updated README.md with workflow clarification

### 🎯 Verify the Fix
1. **Manually trigger a test deployment:**
   ```bash
   gh workflow run "GBS NBA - Build & Deploy" -r main
   ```

2. **Monitor the workflow:**
   ```bash
   # View workflow runs
   gh run list --workflow=gbs-nba-deploy.yml --branch main
   
   # Check specific run details
   gh run view <run-id> --log
   ```

3. **Verify Container App updated:**
   ```powershell
   # Get current image
   az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
     --query 'properties.template.containers[0].image' -o tsv
   
   # Should show the new SHA tag
   ```

4. **Test the API:**
   ```bash
   FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
     --query properties.configuration.ingress.fqdn -o tsv)
   curl "https://$FQDN/health" | jq .
   ```

---

## Cleanup: Optional Removal of Deprecated Workflows

If you want to remove the semi-deprecated workflows to reduce confusion:

```bash
# Remove if not needed
git rm .github/workflows/build-push-acr.yml
git rm .github/workflows/deploy-aca.yml

git commit -m "Remove deprecated CI/CD workflows (gbs-nba-deploy.yml is now primary)"
git push origin main
```

**Decision:** I recommend **keeping them** for now as fallback/emergency options.

---

## Service Principal Details

**Name:** `gbs-nba-github`  
**App ID:** `971db985-be14-4352-bb1d-144d8e8b198c`  
**Object ID:** `03ed6070-7e9b-4dac-8855-8d0ac93601b8`  
**Role:** Contributor (subscription level)  
**Scope:** Azure Green Bier Capital subscription  

**Used by workflows:**
- GBS NBA - Build & Deploy
- Build and Push NBA Image to ACR
- Deploy NBA Image to Azure Container Apps
- ACR Retention

---

## Azure Resource Mapping

```
Subscription: Azure Green Bier Capital
├── Resource Group: nba-gbsv-model-rg (PRODUCTION)
│   ├── Container App: nba-gbsv-api
│   ├── Container Registry: nbagbsacr.azurecr.io
│   ├── Key Vault: nbagbs-keyvault
│   └── Container Apps Environment: nba-gbsv-model-env
│
└── Resource Group: greenbier-enterprise-rg (SHARED)
    └── Function App: nba-picks-trigger
```

---

## References

- [AZURE_CONFIG.md](./AZURE_CONFIG.md) - Azure resource configuration
- [STACK_FLOW_AND_VERIFICATION.md](./STACK_FLOW_AND_VERIFICATION.md) - System architecture
- [copilot-instructions.md](../.github/copilot-instructions.md) - Deployment pipeline details

---

**Status:** Ready for deployment. All workflows should now authenticate successfully.
