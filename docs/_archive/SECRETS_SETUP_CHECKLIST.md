# Secrets Setup Checklist

**Last Updated:** 2026-01-23  
**Status:** Cleaned and updated - OIDC authentication enabled

---

## ✅ Required GitHub Actions Secrets

For CI/CD deployment, configure these in **GitHub → Settings → Secrets and variables → Actions**:

### Core API Keys
| Secret Name | Description | Required |
|-------------|-------------|----------|
| `THE_ODDS_API_KEY` | The Odds API key | ✅ Yes |
| `API_BASKETBALL_KEY` | API-Basketball key | ✅ Yes |

### Azure Authentication (OIDC)
| Secret Name | Description | Required |
|-------------|-------------|----------|
| `AZURE_CLIENT_ID` | Azure Service Principal Client ID | ✅ Yes |
| `AZURE_TENANT_ID` | Azure Tenant ID | ✅ Yes |
| `AZURE_SUBSCRIPTION_ID` | Azure Subscription ID | ✅ Yes |

**Note:** Legacy `AZURE_CREDENTIALS` secret is no longer used. All workflows now use OIDC authentication.

---

## 🔲 Optional GitHub Actions Secrets

### Action Network (Not Currently Used)
| Secret Name | Description |
|-------------|-------------|
| `ACTION_NETWORK_USERNAME` | Action Network username (premium data source) |
| `ACTION_NETWORK_PASSWORD` | Action Network password (premium data source) |

**Status:** These are referenced in code but not actively used. Only set if you plan to enable Action Network integration.

---

## 🔲 Local Development Secrets

For local Docker Compose and development:

### Option 1: Create from Template
```bash
cp .env.example .env
# Edit .env with your actual values
```

### Option 2: Use Docker Secrets
```bash
python scripts/manage_secrets.py create-from-env
```

---

## 🔲 Azure Key Vault Secrets (Production)

**Container App Runtime:** The production Container App reads secrets from Azure Key Vault.

### Required Secrets
1. Go to: https://portal.azure.com → Key vaults → `nbagbs-keyvault`
2. Under "Secrets", ensure these exist:
   - `THE-ODDS-API-KEY` (hyphenated, as Key Vault requires)
   - `API-BASKETBALL-KEY` (hyphenated, as Key Vault requires)

### Via Azure CLI (if you have permissions)
```bash
az keyvault secret set --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --value 'your_key_here'
az keyvault secret set --vault-name nbagbs-keyvault --name API-BASKETBALL-KEY --value 'your_key_here'
```

### Verification
After updating Key Vault, Container App will automatically pick up the new values (may require restart):
```bash
az containerapp restart -n nba-gbsv-api -g nba-gbsv-model-rg
```
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
