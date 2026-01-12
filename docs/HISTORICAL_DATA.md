# Historical Odds Data (Azure-Only)

Historical odds/exports live exclusively in Azure Blob Storage (`nbagbsvstrg` / container `nbahistoricaldata` / prefix `historical/`). Nothing is kept or committed locally.

## Structure in Azure
```
historical/
├── the_odds/
│   ├── events/{season}/events_YYYY-MM-DD.json
│   ├── odds/{season}/odds_YYYY-MM-DD_featured.json
│   ├── period_odds/{season}/period_odds_1h.json
│   └── metadata/
├── exports/
│   ├── {season}_events.(csv|parquet)
│   ├── {season}_odds_featured.(csv|parquet)
│   └── manifest.json
└── derived/elo/... (as produced)
```

## Working locally (temp only)
1) Download what you need:
```
az storage blob download-batch --account-name nbagbsvstrg --auth-mode login \
  --source nbahistoricaldata --destination data/historical --pattern "historical/*"
```
2) Run ingestion/export/backtest.
3) Delete `data/historical` after the run to avoid stale data.

## Ingestion/export scripts
- Continue to use existing scripts (`ingest_historical_period_odds.py`, `export_historical_odds.py`, etc.).
- After generating outputs, upload to Azure with manifests/hashes.
- Do not store historical data in git; Azure Blob is the only source of truth.
