# Infrastructure guide (NBA + shared)

## Layout
- `infra/shared/main.bicep` – one-time shared stack (ACR, Key Vault, Log Analytics, App Insights, Container Apps env).
- `infra/nba/main.bicep` – NBA workload (storage + Container App) built with reusable modules.
- `infra/modules/` – storage + container app modules used by stacks.
- `infra/baseline/` – snapshots from `scripts/export_rg_baseline.ps1`.
- `infra/TAG_POLICY.md` – required tag schema.

## Deploy (manual)
```
# Shared (once)
az deployment group create -g <shared-rg> -f infra/shared/main.bicep

# NBA
az deployment group create -g <nba-rg> -f infra/nba/main.bicep `
  -p theOddsApiKey=<secret> apiBasketballKey=<secret> `
     imageTag=$(Get-Content VERSION)
```

## CI/CD (GitHub Actions)
- Workflow: `.github/workflows/iac.yml`
- Steps: checkout → version read (`VERSION`) → `az bicep build` → `what-if` on PRs → `create` on `main`.
- Required secrets: `AZURE_CREDENTIALS`, `AZURE_RG_NBA`, `AZURE_LOCATION`, `THE_ODDS_API_KEY`, `API_BASKETBALL_KEY`, optional `DATABASE_URL`, `APPINSIGHTS_CONNECTION_STRING`, `WEBSITE_DOMAIN`.

## Baseline + compliance
- Export current state: `powershell -File scripts/export_rg_baseline.ps1 -ResourceGroupName <rg>`
- Audit tags: `powershell -File scripts/rg_compliance_report.ps1 -ResourceGroupName <rg>`
- Commit the latest snapshot before structural changes to preserve history.
