# Secrets Setup Checklist

**Status: Codespace secrets are now set locally. Azure Key Vault needs manual update.**

## ✅ DONE: Local Codespace Secrets

All required secrets have been written to `.env` and `secrets/` in this Codespace:
- `THE_ODDS_API_KEY` = 4a0b80471d1ebeeb74c358fa0fcc4a27
- `API_BASKETBALL_KEY` = eea8757fae3c507add2df14800bae25f
- `ACTION_NETWORK_USERNAME` = jb@greenbiercapital.com
- `ACTION_NETWORK_PASSWORD` = 6nRC!d!Axt3!4nKQ

**Validation passed.** These work immediately in this Codespace.

---

## 🔲 TODO: GitHub Codespaces Secrets (Optional but Recommended)

For **persistent, automatic sync** across all future Codespaces, add these secrets at:
**GitHub → Settings → Codespaces → Repository secrets for `JDSB123/green_bier_sports_nba_model`**

| Secret Name | Value |
|-------------|-------|
| `THE_ODDS_API_KEY` | `4a0b80471d1ebeeb74c358fa0fcc4a27` |
| `API_BASKETBALL_KEY` | `eea8757fae3c507add2df14800bae25f` |
| `ACTION_NETWORK_USERNAME` | `jb@greenbiercapital.com` |
| `ACTION_NETWORK_PASSWORD` | `6nRC!d!Axt3!4nKQ` |

**Why:** Post-create auto-syncs Codespaces secrets → `.env`/`secrets/`. Without this, you'll need to recopy secrets manually when creating new Codespaces.

---

## 🔲 TODO: Azure Key Vault Secrets (Production)

**Manual action required:** The Azure CLI in this Codespace doesn't have Key Vault write permissions.

### Option 1: Azure Portal (Easiest)
1. Go to: https://portal.azure.com → Key vaults → `nbagbs-keyvault`
2. Under "Secrets", add/update:
   - `THE-ODDS-API-KEY` = `4a0b80471d1ebeeb74c358fa0fcc4a27`
   - `API-BASKETBALL-KEY` = `eea8757fae3c507add2df14800bae25f`
   - `ACTION-NETWORK-USERNAME` = `jb@greenbiercapital.com`
   - `ACTION-NETWORK-PASSWORD` = `6nRC!d!Axt3!4nKQ`

### Option 2: Azure CLI (if you have local Azure CLI with Key Vault contributor role)
```bash
az keyvault secret set --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --value '4a0b80471d1ebeeb74c358fa0fcc4a27'
az keyvault secret set --vault-name nbagbs-keyvault --name API-BASKETBALL-KEY --value 'eea8757fae3c507add2df14800bae25f'
az keyvault secret set --vault-name nbagbs-keyvault --name ACTION-NETWORK-USERNAME --value 'jb@greenbiercapital.com'
az keyvault secret set --vault-name nbagbs-keyvault --name ACTION-NETWORK-PASSWORD --value '6nRC!d!Axt3!4nKQ'
```

### Verify Container App References Key Vault
Ensure the Container App (`nba-gbsv-api`) env vars reference Key Vault, NOT plain values:
```bash
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query "properties.template.containers[0].env" -o table
```

Expected format: `@Microsoft.KeyVault(SecretUri=https://nbagbs-keyvault.vault.azure.net/secrets/THE-ODDS-API-KEY)`

**If plain values exist, remove them and use Key Vault references only.**

---

## 🔲 TODO: Remove Duplicate/Legacy Secrets

### Audit and Clean
- **Container App env vars:** Remove any plain-text duplicates; use Key Vault refs only.
- **GitHub repo secrets:** If there are old GitHub Actions secrets not in use, delete them.
- **Old local copies:** If you have an outdated local C: drive folder, archive or delete it to avoid confusion.

---

## Summary

| Location | Status | Action |
|----------|--------|--------|
| **This Codespace** | ✅ Done | Secrets in `.env` and `secrets/` |
| **GitHub Codespaces Secrets** | ⚠️ Manual | Add at GitHub Settings → Codespaces |
| **Azure Key Vault** | ⚠️ Manual | Add via Portal or CLI with proper permissions |
| **Container App** | ⚠️ Audit | Ensure Key Vault refs, remove plain values |

Once all are set, you'll have:
- **Dev (Codespace):** Auto-synced from GitHub Codespaces secrets or local files
- **Production (Azure):** Pulled from Key Vault
- **No confusion:** Single source of truth per environment, no duplicates
