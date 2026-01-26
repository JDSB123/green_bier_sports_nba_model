# Tag policy (single source of truth)

Required tags (applied in Bicep modules):
- `enterprise`: `green-bier-sports-ventures`
- `app`: logical app id (shared or sport-specific, e.g., `nba-model`, `shared-platform`)
- `environment`: `dev|staging|prod`
- `owner`: accountable team (e.g., `sports-analytics`, `platform-eng`)
- `cost_center`: cost tracking (e.g., `sports-nba`, `platform-shared`)
- `compliance`: data/process requirement (e.g., `internal`, `pii-none`)
- `version`: semantic version sourced from repo/pipeline (e.g., `NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD>`)
- `managedBy`: deployment mechanism (`bicep`)

Optional tags (merged via `extraTags`):
- `workload`: scenario or feature flag
- `run_id`: pipeline run or release identifier
- `owner_email`: contact address

Defaults are defined in:
- Shared stack: `infra/shared/main.bicep`
- NBA stack: `infra/nba/main.bicep`

How version flows:
1) `VERSION` file (repo) -> pipeline reads and injects `imageTag`.
2) Bicep modules propagate `version` tag to all resources and set env vars.
3) Deployment outputs reflect the exact version applied.

Validation:
- CI `what-if` ensures tags are present before deploy.
- `scripts/rg_compliance_report.ps1` can flag missing/incorrect tags.
