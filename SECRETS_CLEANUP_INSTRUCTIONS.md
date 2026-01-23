# GitHub Secrets Cleanup - Manual Steps

**Status:** Code changes complete ✅ | Secrets cleanup pending ⏳

The code has been updated to use OIDC authentication and remove all references to deprecated secrets. Now you need to manually remove the old secrets from GitHub.

---

## ✅ Already Removed (Not Found)

These were already cleaned up:
- ~~`NBAGBSVAPI_REGISTRY_PASSWORD`~~
- ~~`NBAGBSVAPI_REGISTRY_USERNAME`~~
- ~~`NBAGBSVAPI_AZURE_CLIENT_ID`~~
- ~~`NBAGBSVAPI_AZURE_TENANT_ID`~~
- ~~`NBAGBSVAPI_AZURE_SUBSCRIPTION_ID`~~

---

## 🔴 ACTION REQUIRED - Remove These Secrets

Go to: **https://github.com/JDSB123/green_bier_sports_nba_model/settings/secrets/actions**

### 1. Remove Action Network Credentials (Not Used)
- [ ] Delete `ACTION_NETWORK_USERNAME`
- [ ] Delete `ACTION_NETWORK_PASSWORD`

### 2. Test OIDC Deployment First
After pushing the code changes, the next deployment will use OIDC instead of `AZURE_CREDENTIALS`.

**Wait for successful deployment**, then:

### 3. Remove Legacy Azure Credentials (After OIDC Works)
- [ ] Delete `AZURE_CREDENTIALS` (only after confirming OIDC deployment works)

---

## ✅ Required Secrets (Keep These)

These must remain:
- ✅ `THE_ODDS_API_KEY`
- ✅ `API_BASKETBALL_KEY`
- ✅ `AZURE_CLIENT_ID`
- ✅ `AZURE_TENANT_ID`
- ✅ `AZURE_SUBSCRIPTION_ID`

---

## 🚀 Push Changes & Deploy

```bash
# Push the code changes
git push origin main

# Monitor the deployment
gh run watch
```

---

## ✅ Verification Checklist

After deployment succeeds:

1. **Verify OIDC works:**
   ```bash
   # Check deployment logs
   gh run list --limit 1
   gh run view --log
   ```

2. **Test the API:**
   ```bash
   curl https://nba-gbsv-api.YOUR_DOMAIN.azurecontainerapps.io/health
   ```

3. **Remove legacy credentials** (via GitHub web UI)

4. **Final verification:**
   - Go to: https://github.com/JDSB123/green_bier_sports_nba_model/settings/secrets/actions
   - Confirm only 5 secrets remain (THE_ODDS_API_KEY, API_BASKETBALL_KEY, AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID)

---

## 🆘 Rollback Plan (If Needed)

If OIDC deployment fails:

1. **Restore legacy auth temporarily:**
   - Edit `.github/workflows/deploy.yml`
   - Change back to `creds: ${{ secrets.AZURE_CREDENTIALS }}`
   - Keep `AZURE_CREDENTIALS` secret

2. **Debug OIDC setup:**
   - Verify service principal has correct permissions
   - Check federated credentials in Azure AD
   - See: https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure

---

**Last Updated:** 2026-01-23  
**Next Action:** Push code, monitor deployment, then clean secrets via web UI
