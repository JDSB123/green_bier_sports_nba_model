# Historical Data: Single Source of Truth (Azure)

## Canonical storage
- Account: `nbagbsvstrg`
- Container: `nbahistoricaldata`
- Prefixes:
  - `historical/` (events, odds, period odds, exports, metadata)
  - `archived_picks/` (by date/version)
  - `models/backtest/{version}/{date}/` (artifacts/results)

## Local usage (ephemeral only)
- Local historical data is not committed to git. Use a temp cache (e.g., `data/historical`) only for a single run, then delete it.
- Pull from Azure before a run; never rely on an old local copy.

## Backtest/run pattern
1) Download from Azure to a temp dir:
   `az storage blob download-batch --account-name nbagbsvstrg --auth-mode login --source nbahistoricaldata --destination data/historical --pattern "historical/*"`
2) Run the job (e.g., `pwsh ./scripts/run_backtest_from_blob.ps1`).
3) Delete the temp dir to prevent reuse.

## Ingestion/export
- Ingestion scripts should write to a temp folder, then upload to Azure with metadata/hashes. Do not persist or commit locally.
- Exports/manifests must be uploaded alongside data to Azure so consumers can validate coverage and integrity.

## Guarantees
- Azure Blob is the only source of truth for historical/backtest data.
- Git stores code + manifests only; no historical datasets.
- Every run fetches fresh from blob; stale/legacy files are removed by deleting the temp cache after use.
