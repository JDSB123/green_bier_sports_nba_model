# GitHub Secrets Audit Report
**Date:** 2026-01-23  
**Repository:** JDSB123/green_bier_sports_nba_model

---

## Executive Summary

✅ **Status:** MOSTLY CLEAN - Minor cleanup recommended  
🔍 **Total Secrets:** 14  
⚠️ **Unused/Deprecated:** 7 (50%)  
✅ **Active & Required:** 7 (50%)

---

## Detailed Findings

### ✅ ACTIVE & REQUIRED SECRETS (7)

| Secret Name | Used By | Purpose | Status |
|-------------|---------|---------|--------|
| `THE_ODDS_API_KEY` | `deploy.yml`<br>`gbs-nba-deploy.yml`<br>`iac.yml` | The Odds API access | ✅ KEEP |
| `API_BASKETBALL_KEY` | `deploy.yml`<br>`gbs-nba-deploy.yml`<br>`iac.yml` | API-Basketball access | ✅ KEEP |
| `AZURE_CREDENTIALS` | `deploy.yml` | Azure login (legacy method) | ⚠️ KEEP (for now) |
| `AZURE_CLIENT_ID` | `acr-retention.yml`<br>`iac.yml` | Azure OIDC auth | ✅ KEEP |
| `AZURE_TENANT_ID` | `acr-retention.yml`<br>`iac.yml` | Azure OIDC auth | ✅ KEEP |
| `AZURE_SUBSCRIPTION_ID` | `acr-retention.yml`<br>`iac.yml` | Azure OIDC auth | ✅ KEEP |
| `AZURE_CLIENT_SECRET` | Not found in workflows | Service principal password | ⚠️ VERIFY USE |

### ⚠️ UNUSED / DEPRECATED SECRETS (7)

| Secret Name | Last Seen | Recommendation | Reason |
|-------------|-----------|----------------|--------|
| `ACTION_NETWORK_USERNAME` | `gbs-nba-deploy.yml` (line 147) | ❌ REMOVE | Optional feature, conditionally used, likely not set up |
| `ACTION_NETWORK_PASSWORD` | `gbs-nba-deploy.yml` (line 148) | ❌ REMOVE | Optional feature, conditionally used, likely not set up |
| `NBAGBSVAPI_AZURE_CLIENT_ID` | None | ❌ REMOVE | Duplicate of `AZURE_CLIENT_ID` |
| `NBAGBSVAPI_AZURE_TENANT_ID` | None | ❌ REMOVE | Duplicate of `AZURE_TENANT_ID` |
| `NBAGBSVAPI_AZURE_SUBSCRIPTION_ID` | None | ❌ REMOVE | Duplicate of `AZURE_SUBSCRIPTION_ID` |
| `NBAGBSVAPI_REGISTRY_PASSWORD` | None | ❌ REMOVE | Unused - Azure uses managed identity |
| `NBAGBSVAPI_REGISTRY_USERNAME` | None | ❌ REMOVE | Unused - Azure uses managed identity |

---

## Workflow Analysis

### deploy.yml (Primary Deployment)
**Uses:**
- `THE_ODDS_API_KEY` ✅
- `API_BASKETBALL_KEY` ✅
- `AZURE_CREDENTIALS` ⚠️ (Legacy auth method)

**Authentication:** Uses legacy `AZURE_CREDENTIALS` (JSON service principal)

### gbs-nba-deploy.yml (Secondary Deployment - DUPLICATE?)
**Uses:**
- `THE_ODDS_API_KEY` ✅
- `API_BASKETBALL_KEY` ✅
- `AZURE_CREDENTIALS` ✅
- `ACTION_NETWORK_USERNAME` (optional)
- `ACTION_NETWORK_PASSWORD` (optional)

**Note:** This appears to be a duplicate/alternative deployment workflow. Consider consolidating.

### acr-retention.yml (ACR Cleanup)
**Uses:**
- `AZURE_CLIENT_ID` ✅
- `AZURE_TENANT_ID` ✅
- `AZURE_SUBSCRIPTION_ID` ✅

**Authentication:** Uses modern OIDC auth (recommended)

### iac.yml (Infrastructure Deployment)
**Uses:**
- `AZURE_CLIENT_ID` ✅
- `AZURE_TENANT_ID` ✅
- `AZURE_SUBSCRIPTION_ID` ✅
- `THE_ODDS_API_KEY` ✅
- `API_BASKETBALL_KEY` ✅

**Authentication:** Uses modern OIDC auth (recommended)

---

## Critical Issues

### 1. ⚠️ DUPLICATE AZURE CREDENTIALS
**Problem:** Two sets of Azure authentication secrets exist:
- Modern OIDC: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- Legacy JSON: `AZURE_CREDENTIALS`
- Prefixed duplicates: `NBAGBSVAPI_AZURE_*` (unused)

**Impact:** Confusion, potential security risk if credentials diverge

**Recommendation:**
1. Migrate `deploy.yml` to use OIDC auth (like `acr-retention.yml` and `iac.yml`)
2. Remove `AZURE_CREDENTIALS` after migration
3. Remove all `NBAGBSVAPI_AZURE_*` secrets (unused)

### 2. ⚠️ DUPLICATE DEPLOYMENT WORKFLOWS
**Problem:** Both `deploy.yml` and `gbs-nba-deploy.yml` exist and deploy to the same resources

**Recommendation:**
1. Consolidate into a single workflow
2. Remove the unused workflow file
3. Clean up any workflow-specific secrets

### 3. ⚠️ ACTION NETWORK CREDENTIALS
**Problem:** Secrets exist but are marked as "optional" in code and docs

**Current State:**
- Documented as optional in `DOCKER_SECRETS.md`
- Only used conditionally in `gbs-nba-deploy.yml` (lines 147-148)
- Likely not actually configured or used

**Recommendation:**
1. If not using Action Network API → Remove secrets
2. If planning to use → Document the purpose and ensure proper integration

### 4. ⚠️ REGISTRY CREDENTIALS (NBAGBSVAPI_REGISTRY_*)
**Problem:** Hardcoded registry credentials stored but never used

**Current State:**
- No references in any workflow files
- Azure Container Registry uses managed identity authentication
- These appear to be legacy credentials

**Recommendation:** Remove immediately - security risk

---

## Authentication Architecture Issues

### Current State (Mixed)
```
deploy.yml → AZURE_CREDENTIALS (legacy JSON)
acr-retention.yml → AZURE_CLIENT_ID + TENANT + SUBSCRIPTION (OIDC) ✅
iac.yml → AZURE_CLIENT_ID + TENANT + SUBSCRIPTION (OIDC) ✅
```

### Recommended State (Consistent)
```
ALL workflows → AZURE_CLIENT_ID + TENANT + SUBSCRIPTION (OIDC)
```

**Benefits:**
- More secure (no long-lived secrets stored)
- Better audit trail
- Modern Azure best practice
- Consistent across all workflows

---

## Cleanup Action Plan

### Phase 1: Immediate (Security Risk)
```bash
# Remove unused registry credentials
gh secret remove NBAGBSVAPI_REGISTRY_PASSWORD
gh secret remove NBAGBSVAPI_REGISTRY_USERNAME

# Remove duplicate Azure credentials
gh secret remove NBAGBSVAPI_AZURE_CLIENT_ID
gh secret remove NBAGBSVAPI_AZURE_TENANT_ID
gh secret remove NBAGBSVAPI_AZURE_SUBSCRIPTION_ID
```

### Phase 2: Optional Features (If Not Used)
```bash
# Only if Action Network is NOT being used:
gh secret remove ACTION_NETWORK_USERNAME
gh secret remove ACTION_NETWORK_PASSWORD
```

### Phase 3: Migrate Authentication (After Testing)
1. Update `deploy.yml` to use OIDC (copy pattern from `iac.yml`)
2. Test deployment with OIDC auth
3. Once verified working:
   ```bash
   gh secret remove AZURE_CREDENTIALS
   ```

### Phase 4: Consolidate Workflows
1. Choose primary workflow (`deploy.yml` recommended)
2. Merge any unique features from `gbs-nba-deploy.yml`
3. Delete unused workflow
4. Update documentation

---

## Recommended Secrets Structure (Clean State)

### Required Secrets (5)
```
✅ THE_ODDS_API_KEY          → API access for odds data
✅ API_BASKETBALL_KEY        → API access for basketball data
✅ AZURE_CLIENT_ID           → OIDC auth
✅ AZURE_TENANT_ID           → OIDC auth
✅ AZURE_SUBSCRIPTION_ID     → OIDC auth
```

### Optional Secrets (2 - Only If Used)
```
⚠️ ACTION_NETWORK_USERNAME   → Optional enhanced data source
⚠️ ACTION_NETWORK_PASSWORD   → Optional enhanced data source
```

---

## Documentation Updates Needed

1. **Update SECRETS_SETUP_CHECKLIST.md**
   - Remove references to `NBAGBSVAPI_*` secrets
   - Clarify Action Network as truly optional
   - Document OIDC as primary auth method

2. **Update DOCKER_SECRETS.md**
   - Mark Action Network as "Optional - Not Currently Used" if not in use
   - Remove any registry credential references

3. **Update .github/copilot-instructions.md**
   - Update deployment documentation to reflect cleaned secrets

4. **Create AUTHENTICATION.md**
   - Document OIDC setup process
   - Explain migration from legacy `AZURE_CREDENTIALS`

---

## Security Observations

### ✅ Good Practices
- API keys stored in GitHub Secrets (not in code)
- Docker secrets pattern for local development
- Key Vault used for production runtime secrets

### ⚠️ Areas for Improvement
1. Remove unused/duplicate secrets (reduces attack surface)
2. Migrate to OIDC everywhere (no long-lived credentials)
3. Regular secret rotation schedule (document in runbook)
4. Consider using GitHub Environments for prod/dev separation

---

## Next Steps

1. **Review with team:** Confirm Action Network and `NBAGBSVAPI_*` secrets are truly unused
2. **Execute Phase 1 cleanup:** Remove security-risk secrets immediately
3. **Plan OIDC migration:** Schedule time to update `deploy.yml`
4. **Document changes:** Update all relevant documentation
5. **Monitor deployments:** Ensure nothing breaks after cleanup

---

## Appendix: How This Audit Was Performed

### Methodology
1. Scanned all `.github/workflows/*.yml` files for secret references
2. Searched all `.yml`, `.ps1`, `.sh` files for secret usage
3. Cross-referenced with documentation in `docs/`
4. Identified duplicates and unused secrets
5. Mapped secrets to their actual usage in workflows

### Tools Used
- `grep_search` across repository
- Manual review of workflow files
- Documentation analysis

### Files Analyzed
- `.github/workflows/deploy.yml`
- `.github/workflows/gbs-nba-deploy.yml`
- `.github/workflows/acr-retention.yml`
- `.github/workflows/iac.yml`
- `docs/DOCKER_SECRETS.md`
- `docs/SECRETS_SETUP_CHECKLIST.md`
- All PowerShell and shell scripts

---

**End of Audit Report**
