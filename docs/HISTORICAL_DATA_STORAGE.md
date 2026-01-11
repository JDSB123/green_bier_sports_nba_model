# Historical Data Storage Guide

This document describes the system for storing historical NBA data, backtesting models, and archived picks in both a dedicated git repository and Azure Blob Storage.

## Overview

The historical data storage system consists of:

1. **Git Repository**: `nba-historical-data` - Stores historical data that is committed to git (not ignored)
2. **Azure Blob Storage**: Container `nbahistoricaldata` in storage account `nbagbsvstrg`
3. **Versioning**: All data includes timestamps and model versions to prevent overwrites

## Data Ingestion Workflow

### Raw Data Sources
- **The Odds API** | primary: spreads, totals, betting splits, and event metadata that power FG + 1H markets.
- **API-Basketball** | secondary: canonical game outcomes, team stats, box scores, and schedules when The Odds API lacks coverage.
- **Kaggle / NCAA dumps** | archival snapshots used to seed seasons and cross-check derived metrics for integrity.
- **Open-source GitHub feeds** | curated exports (schedules, advanced stats) that fill gaps or provide alternate naming references.
- **NCAAr institutional files** | localized downloads ingested alongside the other sources to keep season coverage complete.

### Single Source of Storage Truth
- Every NBA ingestion pipeline writes its raw snapshot and normalized export to Azure Blob Storage before any downstream logic runs: `nbahistoricaldata` inside `nbagbsvstrg`/`nba-gbsv-model-rg`.
- Historical NBA backtests, production models, and QA tooling read directly from that container (or the synced git repo) so all scripts reference the same canonical dataset. No additional copies are created that could drift.
- Each upload includes metadata (source, season, timestamp, manifest hash) so downstream tooling can prove which blob/backtest file corresponds to a specific NBA season/version.

### Raw → Standardized Source of Truth
- Raw files are stored season-by-season (2023-2024, 2024-2025, 2025-2026) and truncated to `today() - 3 days` to prevent leakage into live predictions/backtesting.
- Before merging, every source is canonicalized: team-name variants map to the single-purpose team variant database (add only new teams/variants, no wholesale rewrites), timestamps convert to CST, and match keys stay consistent.
- Season coverage is annotated. If a source offers only partial data for a given season (e.g., missing 1H odds), the manifest records the gap so downstream backtests know to restrict the metrics used and avoid leakage.
- After standardization the data lands in `data/processed/` and updates `data_manifest.json` with schema, coverage, checksums, and the earliest/latest timestamp.

### QA/QC & Anti-Leakage Controls
- `scripts/verify_data_standardization.py` and `scripts/validate_training_data.py` enforce canonical team names, schema expectations, checksums, and row counts before any backtest or training job consumes the files.
- Backtesting reads the latest standardized dataset, but only up to `today() - 3 days` so models never see future games. If any season/source lacks complete data, the manifest flags it and the backtest falls back to fewer features for that period.
- Each historical backtest run is anchored to this standardized, single-source dataset—no staging copies or ad hoc merges are permitted—so version control and anti-leakage rules remain intact.

### Endpoint Coverage (Prod vs Historical)
- **Production/predictions:** Use the comprehensive ingestion pipeline (`scripts/ingest_all.py` → `ComprehensiveIngestion`) so ALL live endpoints are hit: The Odds API (events, FG/1H/Q1 odds, participants, betting splits), API-Basketball (teams/games/stats/standings), ESPN injuries (plus API-Basketball injuries when keys exist), Action Network splits, and BetsAPI live/alt odds where available.
- **Historical backtesting:** Use only stored, time-bound datasets (The Odds historical/derived lines, API/NBA box scores, Kaggle/archival feeds). Betting splits are not available historically; they are excluded from backtests, and manifests must note that absence so feature expectations stay consistent.
- **Policy:** If an endpoint is unavailable for a season (e.g., splits historically), record the gap in the manifest and proceed with the remaining features; do not silently inject placeholders.
- **Blob as single source:** Run builders with `--sync-from-azure` (e.g., `build_training_data_complete.py`, `merge_training_data.py`) to pull odds/exports/raw feeds from `nbahistoricaldata` before processing. No local-only historical snapshots should be assumed; everything should originate from the blob, and manifests should reflect what was actually present there.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NBA Model Repository                      │
│              (green_bier_sports_nba_model)                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Production Model & Live Predictions                 │   │
│  │  - Ignores historical data in git                   │   │
│  │  - Focuses on current predictions                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Archives picks & data
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              NBA Historical Data Repository                  │
│                  (nba-historical-data)                      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Historical Data (COMMITTED to git)                  │   │
│  │  - Historical odds & events                          │   │
│  │  - Archived picks with timestamps                    │   │
│  │  - Backtest models & results                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Syncs to
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Azure Blob Storage                             │
│  Container: nbahistoricaldata                              │
│  Storage Account: nbagbsvstrg                              │
│  Resource Group: nba-gbsv-model-rg                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Organized by:                                        │   │
│  │  - Date/Season                                       │   │
│  │  - Model Version                                     │   │
│  │  - Data Type                                         │   │
│  │  - Timestamp (prevents overwrites)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Repository Structure

### nba-historical-data Repository

```
nba-historical-data/
├── data/
│   ├── raw/                          # Raw ingested data
│   ├── processed/                    # Processed/cleaned data
│   ├── historical/                  # Historical odds and events
│   │   ├── the_odds/                # Raw JSON from The Odds API
│   │   │   ├── events/              # Historical events by season
│   │   │   ├── odds/                # Historical odds snapshots
│   │   │   ├── period_odds/         # Historical period odds
│   │   │   ├── player_props/        # Player props
│   │   │   └── metadata/            # Ingestion tracking
│   │   └── exports/                 # Normalized CSV/Parquet exports
│   └── archived_picks/              # Archived picks
│       ├── by_date/                 # Organized by date
│       └── by_version/              # Organized by model version
├── models/
│   ├── backtest/                    # Backtesting model artifacts
│   └── fine_tuned/                  # Fine-tuned model versions
├── backtest_results/                # Backtest analysis results
│   ├── by_season/                   # Results by season
│   └── by_market/                   # Results by market
├── scripts/                         # Utility scripts
└── docs/                            # Documentation
```

### Azure Blob Storage Structure

```
nbahistoricaldata/
├── archived_picks/
│   ├── {YYYY-MM-DD}/                # Picks organized by date
│   │   ├── v{version}_{timestamp}.jsonl
│   │   └── v{version}_{timestamp}_metadata.json
│   └── by_version/
│       └── {version}/
│           └── {YYYY-MM-DD}/
│               └── picks_{timestamp}.jsonl
├── historical/
│   └── the_odds/
│       ├── events/{season}/
│       ├── odds/{season}/
│       ├── period_odds/{season}/
│       ├── player_props/{season}/
│       ├── metadata/
│       └── exports/
└── models/
    └── backtest/
        └── {version}/
            └── {YYYY-MM-DD}/
                ├── models/
                ├── results/
                └── metadata.json
```

## Setup

### 1. Initialize Historical Data Repository

```powershell
# Run from the NBA model repository root
.\scripts\init_historical_data_repo.ps1

# Or specify custom path
.\scripts\init_historical_data_repo.ps1 -RepoPath "C:\repos\nba-historical-data"
```

This creates:
- Directory structure
- README.md
- .gitignore
- Initial git commit

### 2. Deploy Azure Storage Container

The `nbahistoricaldata` container is automatically created when deploying the Azure infrastructure:

```powershell
# Deploy infrastructure (includes new container)
cd infra/nba
az deployment group create `
    -g nba-gbsv-model-rg `
    -f main.bicep `
    -p theOddsApiKey=... apiBasketballKey=... imageTag=<tag>
```

The container is added to the storage account `nbagbsvstrg` in resource group `nba-gbsv-model-rg`.

## Usage

### Archiving Picks Before Front-End Updates

**When to use**: Before running fresh picks that will overwrite existing data on front ends.

```powershell
# Archive picks for a specific date
.\scripts\archive_picks_to_azure.ps1 -Date "2025-01-15"

# With specific model version
.\scripts\archive_picks_to_azure.ps1 `
    -Date "2025-01-15" `
    -ModelVersion "NBA_v33.0.11.0"
```

**What it does**:
1. Copies picks to local archive: `data/archived_picks/by_date/{date}/`
2. Uploads to Azure with timestamp: `archived_picks/{date}/v{version}_{timestamp}.jsonl`
3. Creates versioned copy: `archived_picks/by_version/{version}/{date}/`
4. Generates metadata file with version and timestamp info

**Output**:
- Local: `data/archived_picks/by_date/2025-01-15/picks_20250115_143022.jsonl`
- Azure: `archived_picks/2025-01-15/vNBA_v33_0_11_0_20250115_143022.jsonl`

### Syncing Historical Data to Azure

**When to use**: After ingesting historical data or periodically to ensure Azure backup.

```powershell
# Sync all historical data
.\scripts\sync_historical_data_to_azure.ps1

# Sync specific season
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"

# Sync specific data type
.\scripts\sync_historical_data_to_azure.ps1 -DataType "odds"

# Dry run (see what would be synced)
.\scripts\sync_historical_data_to_azure.ps1 -DryRun
```

**What it does**:
- Syncs files from `data/historical/` to Azure Blob Storage
- Maintains directory structure
- Preserves all historical data

### Storing Backtest Models

**When to use**: After running backtests to preserve model artifacts for reproducibility.

```powershell
# Store backtest model
.\scripts\store_backtest_model.ps1 `
    -Version "NBA_v33.0.11.0" `
    -BacktestDate "2025-01-15"

# With custom paths
.\scripts\store_backtest_model.ps1 `
    -Version "NBA_v33.0.11.0" `
    -BacktestDate "2025-01-15" `
    -ModelPath "models/production" `
    -ResultsPath "backtest_results"
```

**What it does**:
1. Copies model files and results to local repository
2. Creates metadata file
3. Uploads to Azure Blob Storage with versioning

## Data Organization Principles

### Timestamps
- Format: `YYYYMMDD_HHMMSS` (e.g., `20250115_143022`)
- ISO 8601 format in metadata: `2025-01-15T14:30:22Z`
- Ensures unique file names and prevents overwrites

### Versioning
- Model versions: `NBA_v33.0.11.0`
- Safe version strings: `NBA_v33_0_11_0` (replaces special chars)
- Versioned paths: `by_version/{version}/{date}/`

### Naming Conventions
- Picks: `picks_{timestamp}.jsonl`
- Models: `models/backtest/{version}/{date}/`
- Metadata: `*_metadata.json`

## Workflow Examples

### Daily Picks Archival Workflow

```powershell
# 1. Generate fresh picks (this will overwrite front-end data)
.\scripts\run_slate.py --date today

# 2. Archive picks BEFORE they're overwritten
.\scripts\archive_picks_to_azure.ps1 -Date (Get-Date -Format "yyyy-MM-dd")

# 3. Verify archive
az storage blob list `
    --account-name nbagbsvstrg `
    --container-name nbahistoricaldata `
    --prefix "archived_picks/$(Get-Date -Format 'yyyy-MM-dd')" `
    --output table
```

### Historical Data Ingestion Workflow

```powershell
# 1. Ingest historical data (saves locally)
python scripts/ingest_historical_odds.py --season 2024-2025

# 2. Sync to Azure for backup
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"

# 3. Commit to git repository
cd ../nba-historical-data
git add data/historical/
git commit -m "Add 2024-2025 historical odds data"
git push
```

### Backtest Model Storage Workflow

```powershell
# 1. Run backtest
docker compose -f docker-compose.backtest.yml up

# 2. Store model artifacts
.\scripts\store_backtest_model.ps1 `
    -Version "NBA_v33.0.11.0" `
    -BacktestDate (Get-Date -Format "yyyy-MM-dd")

# 3. Commit to git repository
cd ../nba-historical-data
git add models/backtest/
git commit -m "Store backtest model NBA_v33.0.11.0"
git push
```

## Azure Storage Access

### List Archived Picks

```powershell
# List picks for a date
az storage blob list `
    --account-name nbagbsvstrg `
    --container-name nbahistoricaldata `
    --prefix "archived_picks/2025-01-15" `
    --output table

# List by version
az storage blob list `
    --account-name nbagbsvstrg `
    --container-name nbahistoricaldata `
    --prefix "archived_picks/by_version/NBA_v33_0_11_0" `
    --output table
```

### Download Archived Data

```powershell
# Download picks
az storage blob download `
    --account-name nbagbsvstrg `
    --container-name nbahistoricaldata `
    --name "archived_picks/2025-01-15/vNBA_v33_0_11_0_20250115_143022.jsonl" `
    --file "downloaded_picks.jsonl"
```

### Get Storage Account Connection String

```powershell
az storage account show-connection-string `
    --name nbagbsvstrg `
    --resource-group nba-gbsv-model-rg `
    --output tsv
```

## Best Practices

1. **Always Archive Before Overwrites**: Archive picks before running fresh predictions that overwrite front-end data.

2. **Version Everything**: Include model version and timestamp in all archived data.

3. **Regular Syncs**: Periodically sync historical data to Azure for backup.

4. **Git Commits**: Commit historical data to the git repository for version control.

5. **Metadata**: Always include metadata files with timestamps, versions, and source information.

6. **Organization**: Follow the directory structure for consistency.

7. **Dry Runs**: Use `-DryRun` flag to preview operations before executing.

## Troubleshooting

### Azure CLI Not Found
```powershell
# Install Azure CLI
winget install -e --id Microsoft.AzureCLI
```

### Storage Account Access Denied
- Verify you're logged in: `az account show`
- Check permissions: `az role assignment list --assignee <your-email>`
- Ensure you have Storage Blob Data Contributor role

### Container Not Found
- Verify container exists: `az storage container list --account-name nbagbsvstrg`
- Deploy infrastructure if missing: `cd infra/nba; az deployment group create ...`

### File Not Found Errors
- Check paths are relative to repository root
- Verify data exists: `Test-Path data/historical/the_odds/events`

## Related Documentation

- [Historical Data Documentation](./HISTORICAL_DATA.md) - Historical data ingestion
- [Azure Configuration](./AZURE_CONFIG.md) - Azure infrastructure setup
- [Data Ingestion Methodology](./DATA_INGESTION_METHODOLOGY.md) - Data ingestion processes

## Support

For issues or questions:
1. Check Azure storage account logs
2. Review script error messages
3. Verify Azure CLI connectivity
4. Check file permissions and paths
