# Historical Data Storage Guide

Historical datasets, backtest artifacts, and archived picks have a **single source of truth in Azure Blob Storage**. Local copies under `data/historical/` are working caches only and are now gitignored. Nothing in this repository should be treated as canonical historical data.

## Canonical location
- Storage account: `nbagbsvstrg` (resource group `nba-gbsv-model-rg`)
- Container: `nbahistoricaldata`
- Access: Azure CLI login (`az login`) or managed identity with Blob Data Contributor/Reader

## Layout in Azure
```
nbahistoricaldata/
├── archived_picks/
│   ├── {YYYY-MM-DD}/v{version}_{timestamp}.jsonl
│   └── by_version/{version}/{YYYY-MM-DD}/picks_{timestamp}.jsonl
├── historical/
│   └── the_odds/
│       ├── events/{season}/
│       ├── odds/{season}/
│       ├── period_odds/{season}/
│       ├── player_props/{season}/
│       ├── metadata/
│       └── exports/
└── models/
    └── backtest/{version}/{YYYY-MM-DD}/
        ├── models/
        ├── results/
        └── metadata.json
```

## Local working set (gitignored)
- Path: `data/historical/` (mirrors the blob layout)
- Use it as a cache for ingestion/backtests; always pull from/push to Azure to avoid drift.

## Sync workflows
- **Upload local → Azure (overwrite existing)**
  `az storage blob upload-batch --account-name nbagbsvstrg --auth-mode login --destination nbahistoricaldata --source data/historical --destination-path historical --overwrite`

- **Download Azure → local** (refresh your cache)
  `az storage blob download-batch --account-name nbagbsvstrg --auth-mode login --destination data/historical --source nbahistoricaldata --pattern "historical/*"`

- **Archive picks to Azure**
  `.\scripts\archive_picks_to_azure.ps1 -Date "YYYY-MM-DD" [-ModelVersion <version>]`

- **Sync historical data (scripted)**
  `.\scripts\sync_historical_data_to_azure.ps1 [-Season 2024-2025] [-DataType odds] [-DryRun]`

- **Store backtest models/results to Azure**
  `.\scripts\store_backtest_model.ps1 -Version "<tag>" -BacktestDate "YYYY-MM-DD" [-ModelPath ...] [-ResultsPath ...]`

## QA / anti-leakage controls
- Data is season-scoped and truncated to `today() - 3 days` for backtests to prevent future leakage.
- Manifests and metadata must travel with uploads (source, season, timestamp, hash).
- Backtest scripts should read from the blob-synced cache (`--sync-from-azure` where available) instead of local-only files.

## Troubleshooting
- **Historical scripts blocked**: set `HISTORICAL_MODE=true` and `HISTORICAL_OUTPUT_ROOT` to an Azure-mounted path. For local testing only, set `ALLOW_LOCAL_HISTORICAL=true`.
- **Access denied**: verify `az account show` and that you have Blob Data Contributor/Reader on `nbagbsvstrg`.
- **Storage missing**: re-run infra deploy (`infra/nba/prediction.bicep` or `infra/nba/main.bicep`) to recreate the storage account.
- **Large transfers**: add `--max-connections 4` to `az storage blob upload-batch`/`download-batch` if needed.
