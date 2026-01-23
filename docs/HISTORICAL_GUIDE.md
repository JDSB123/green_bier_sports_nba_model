# Historical Data Guide

**Last Updated:** 2026-01-23
**Status:** Consolidated from HISTORICAL_DATA, HISTORICAL_DATA_BEST_PRACTICES, HISTORICAL_DATA_STORAGE

---

## Core Principle

**Historical data lives exclusively in Azure Blob Storage.** Local copies are ephemeral working caches only.

---

## Azure Location

**Storage Account:** `nbagbsvstrg`
**Container:** `nbahistoricaldata`

```
nbahistoricaldata/
├── historical/
│   └── the_odds/
│       ├── events/{season}/events_YYYY-MM-DD.json
│       ├── odds/{season}/odds_YYYY-MM-DD_featured.json
│       ├── period_odds/{season}/period_odds_1h.json
│       └── exports/
├── archived_picks/
│   ├── {YYYY-MM-DD}/v{version}_{timestamp}.jsonl
│   └── by_version/{version}/...
└── models/
    └── backtest/{version}/{YYYY-MM-DD}/
        ├── models/
        ├── results/
        └── metadata.json
```

---

## Environment Guards

Historical scripts require environment variables:

| Variable | Required | Purpose |
|----------|----------|---------|
| `HISTORICAL_MODE=true` | Yes | Enables historical workflows |
| `HISTORICAL_OUTPUT_ROOT` | Yes | Azure-mounted output path |
| `ALLOW_LOCAL_HISTORICAL=true` | No | Override for local testing only |

Without these, historical scripts will fail with a clear error.

---

## Working Locally (Ephemeral Only)

1. **Download what you need:**
```bash
az storage blob download-batch \
  --account-name nbagbsvstrg \
  --auth-mode login \
  --source nbahistoricaldata \
  --destination data/historical \
  --pattern "historical/*"
```

2. **Run your job** (backtest, ingestion, export)

3. **Delete the local cache:**
```bash
rm -rf data/historical
```

Never commit `data/historical/` to git. It's gitignored.

---

## Sync Workflows

### Upload Local → Azure

```bash
az storage blob upload-batch \
  --account-name nbagbsvstrg \
  --auth-mode login \
  --destination nbahistoricaldata \
  --source data/historical \
  --destination-path historical \
  --overwrite
```

### Download Azure → Local

```bash
az storage blob download-batch \
  --account-name nbagbsvstrg \
  --auth-mode login \
  --destination data/historical \
  --source nbahistoricaldata \
  --pattern "historical/*"
```

### PowerShell Scripts

```powershell
# Sync historical data
.\scripts\sync_historical_data_to_azure.ps1 -Season 2024-2025 -DataType odds

# Archive picks
.\scripts\archive_picks_to_azure.ps1 -Date "YYYY-MM-DD"

# Store backtest models
.\scripts\store_backtest_model.ps1 -Version "<tag>" -BacktestDate "YYYY-MM-DD"
```

---

## Historical Scripts

| Script | Purpose |
|--------|---------|
| `historical_ingest_period_odds.py` | Ingest period odds from The Odds API |
| `historical_ingest_elo_ratings.py` | Ingest FiveThirtyEight Elo ratings |
| `historical_export_odds.py` | Export historical odds to CSV |
| `historical_export_period_odds.py` | Export period odds |
| `historical_fetch_quarter_scores.py` | Fetch Q1-Q4 scores |
| `historical_backtest_production.py` | Production backtest |
| `historical_backtest_extended.py` | Extended backtest |
| `historical_audit_data_integrity.py` | Audit data quality |
| `historical_audit_backtest_caching.py` | Audit backtest caching |

All historical scripts:
- Require `HISTORICAL_MODE=true`
- Default output to `HISTORICAL_OUTPUT_ROOT` (Azure path)
- Fail if local paths detected without override

---

## Anti-Leakage Controls

- Data is season-scoped and truncated to `today() - 3 days` for backtests
- Manifests and metadata must travel with uploads
- Every run fetches fresh from blob; stale files removed after use

---

## Troubleshooting

### Historical scripts blocked
Set environment variables:
```bash
export HISTORICAL_MODE=true
export HISTORICAL_OUTPUT_ROOT=/mnt/azure/historical
```

For local testing only:
```bash
export ALLOW_LOCAL_HISTORICAL=true
```

### Access denied
Verify Azure login:
```bash
az account show
```
Ensure Blob Data Contributor/Reader role on `nbagbsvstrg`.

### Storage missing
Re-run infra deploy:
```bash
pwsh ./infra/nba/deploy.ps1
```
