# Infrastructure Guide (NBA + Shared)

## Layout
- `infra/shared/main.bicep` – one-time shared stack (ACR, Key Vault, Log Analytics, App Insights, Container Apps env).
- `infra/nba/main.bicep` – NBA workload (storage + Container App) built with reusable modules.
- `infra/nba/deploy.ps1` – PowerShell wrapper for deploying NBA infrastructure.
- `infra/modules/` – storage + container app modules used by stacks.
- `infra/baseline/` – snapshots from `scripts/export_rg_baseline.ps1`.
- `infra/TAG_POLICY.md` – required tag schema.

## Deploy (manual)

```powershell
# Using deploy script (recommended)
pwsh ./infra/nba/deploy.ps1

# With specific tag
pwsh ./infra/nba/deploy.ps1 -Tag NBA_v33.0.8.0

# Preview changes (what-if)
pwsh ./infra/nba/deploy.ps1 -WhatIf

# Direct az CLI (shared - once)
az deployment group create -g nba-gbsv-model-rg -f infra/shared/main.bicep
```

## CI/CD (GitHub Actions)
- Workflow: `.github/workflows/iac.yml`
- Triggers: Push to `main` (deploy) or PRs (what-if validation)
- Steps: checkout → version read (`VERSION`) → `az bicep build` → `what-if` on PRs → `create` on `main`
- Required secrets (OIDC): `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- App secrets: `THE_ODDS_API_KEY`, `API_BASKETBALL_KEY`

## Baseline + compliance
- Export current state: `powershell -File scripts/export_rg_baseline.ps1 -ResourceGroupName <rg>`
- Audit tags: `powershell -File scripts/rg_compliance_report.ps1 -ResourceGroupName <rg>`
- Commit the latest snapshot before structural changes to preserve history.
