# Infrastructure Guide

## Single Entry Point

All Azure resources are defined in **one file**: `infra/nba/main.bicep`

This deploys everything to `nba-gbsv-model-rg`:

### Platform Layer
- Container Registry (`nbagbsacr`)
- Key Vault (`nbagbs-keyvault`)
- Log Analytics (`gbs-logs-prod`)
- App Insights (`gbs-insights-prod`)

### Data Layer
- Storage Account (`nbagbsvstrg`) with containers: models, predictions, results

### Compute Layer
- Container Apps Environment (`nba-gbsv-model-env`)
- Container App (`nba-gbsv-api`)

### Teams Bot Layer
- App Service Plan (`nba-gbsv-func-plan`) - Consumption/Dynamic
- Function App (`nba-picks-trigger`) - Python 3.11
- Bot Service (`nba-picks-bot`)

## Layout

```
infra/
├── nba/
│   ├── main.bicep      ← Single source of truth (ALL resources)
│   └── deploy.ps1      ← PowerShell wrapper script
├── modules/
│   ├── storage.bicep   ← Reusable storage module
│   └── containerApp.bicep ← Reusable container app module
├── baseline/           ← Snapshots from export script
└── TAG_POLICY.md       ← Required tag schema
```

## Deploy

```powershell
# Using deploy script (recommended)
pwsh ./infra/nba/deploy.ps1

# With specific tag
pwsh ./infra/nba/deploy.ps1 -Tag NBA_v33.0.11.0

# Preview changes (what-if)
pwsh ./infra/nba/deploy.ps1 -WhatIf

# Direct az CLI (full deployment including Teams Bot)
az deployment group create -g nba-gbsv-model-rg -f infra/nba/main.bicep `
  -p theOddsApiKey=<secret> `
     apiBasketballKey=<secret> `
     microsoftAppId=<bot-app-id> `
     microsoftAppTenantId=<tenant-id> `
     microsoftAppPassword=<bot-secret>
```

## CI/CD (GitHub Actions)

- Workflow: `.github/workflows/iac.yml`
- Triggers: Push to `main` (deploy) or PRs (what-if validation)
- Required secrets (OIDC): `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- App secrets: `THE_ODDS_API_KEY`, `API_BASKETBALL_KEY`

## Compliance

```powershell
# Export current state
powershell -File scripts/export_rg_baseline.ps1 -ResourceGroupName nba-gbsv-model-rg

# Audit tags
powershell -File scripts/rg_compliance_report.ps1 -ResourceGroupName nba-gbsv-model-rg
```

