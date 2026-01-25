# Secrets Management

**Last Updated:** 2026-01-23
**Status:** Azure‑first secrets management guide

---

## Required API Keys

| Secret | Description | Required |
|--------|-------------|----------|
| `THE_ODDS_API_KEY` | The Odds API | ✅ Yes |
| `API_BASKETBALL_KEY` | API-Basketball | ✅ Yes |

## Optional Secrets

| Secret | Description | Used |
|--------|-------------|------|
| `ACTION_NETWORK_USERNAME` | Premium betting splits | Required when `REQUIRE_ACTION_NETWORK_SPLITS` or `REQUIRE_REAL_SPLITS` is true |
| `ACTION_NETWORK_PASSWORD` | Premium betting splits | Required when `REQUIRE_ACTION_NETWORK_SPLITS` or `REQUIRE_REAL_SPLITS` is true |

---

## Secret Resolution Order (Azure)

1. Container App secret references (`secretRef`)
2. Environment variables
3. Default (fail if required)

---

## GitHub Actions (CI/CD)

Configure in **GitHub → Settings → Secrets and variables → Actions**:

### Required Secrets

| Secret | Description |
|--------|-------------|
| `THE_ODDS_API_KEY` | The Odds API key |
| `API_BASKETBALL_KEY` | API-Basketball key |
| `AZURE_CLIENT_ID` | Azure SP Client ID (OIDC) |
| `AZURE_TENANT_ID` | Azure Tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Azure Subscription ID |

**Note:** We use OIDC authentication. No `AZURE_CREDENTIALS` secret needed.

---

## Azure Key Vault (Production)

**Key Vault:** `nbagbs-keyvault`

The Container App reads secrets from Key Vault at runtime.

### Required Secrets

| Key Vault Secret | Description |
|------------------|-------------|
| `THE-ODDS-API-KEY` | The Odds API (hyphenated) |
| `API-BASKETBALL-KEY` | API-Basketball (hyphenated) |

### Set via CLI

```bash
az keyvault secret set \
  --vault-name nbagbs-keyvault \
  --name THE-ODDS-API-KEY \
  --value 'your_key_here'

az keyvault secret set \
  --vault-name nbagbs-keyvault \
  --name API-BASKETBALL-KEY \
  --value 'your_key_here'
```

### Verify Container App References

```bash
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query "properties.template.containers[0].env" -o table
```

Expected: `secretRef: the-odds-api-key` (not plain values)

---

## Secret Management Script

```bash
# Create secrets from .env
python scripts/manage_secrets.py create-from-env

# List secrets
python scripts/manage_secrets.py list

# Validate secrets
python scripts/manage_secrets.py validate
```

---

## Never Commit Secrets

- `.env` is gitignored (use `.env.example` as template)
- `secrets/` directory is gitignored
- Never hardcode keys in code

---

## Verification

### Script Validation
```bash
python scripts/manage_secrets.py validate
```

### Container
```bash
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health" | jq '.api_keys'
```

Expected:
```json
{
  "THE_ODDS_API_KEY": "set",
  "API_BASKETBALL_KEY": "set"
}
```

### Production
```bash
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health" | jq '.api_keys'
```
