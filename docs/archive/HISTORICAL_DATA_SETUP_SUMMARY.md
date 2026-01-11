# Historical Data Storage Setup Summary

This document summarizes the setup for the NBA historical data storage system, similar to the NCAAM model/repos pattern.

## What Was Created

### 1. Azure Infrastructure Updates

**File**: `infra/nba/main.bicep`
- Added `nbahistoricaldata` container to the storage account `nbagbsvstrg`
- Container is created automatically when deploying infrastructure
- Located in resource group `nba-gbsv-model-rg`

### 2. Repository Initialization Script

**File**: `scripts/init_historical_data_repo.ps1`
- Creates the `nba-historical-data` git repository structure
- Sets up directory hierarchy for historical data, archived picks, and backtest models
- Initializes git repository with initial commit
- Creates README.md and .gitignore

**Usage**:
```powershell
.\scripts\init_historical_data_repo.ps1
```

### 3. Pick Archival Script

**File**: `scripts/archive_picks_to_azure.ps1`
- Archives picks with unique timestamps and versions before front-end overwrites
- Stores locally in `data/archived_picks/by_date/`
- Uploads to Azure Blob Storage container `nbahistoricaldata`
- Creates versioned copies and metadata files

**Usage**:
```powershell
.\scripts\archive_picks_to_azure.ps1 -Date "2025-01-15"
```

### 4. Historical Data Sync Script

**File**: `scripts/sync_historical_data_to_azure.ps1`
- Syncs historical data from local `data/historical/` to Azure Blob Storage
- Supports filtering by season and data type
- Maintains directory structure in blob storage
- Includes dry-run mode for preview

**Usage**:
```powershell
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"
```

### 5. Backtest Model Storage Script

**File**: `scripts/store_backtest_model.ps1`
- Stores backtest model artifacts to both local repository and Azure
- Includes model files, results, and metadata
- Organizes by model version and backtest date

**Usage**:
```powershell
.\scripts\store_backtest_model.ps1 -Version "NBA_v33.0.11.0" -BacktestDate "2025-01-15"
```

### 6. Documentation

**Files Created**:
- `docs/HISTORICAL_DATA_STORAGE.md` - Comprehensive guide for historical data storage
- `docs/HISTORICAL_DATA_SETUP_SUMMARY.md` - This summary document

**Files Updated**:
- `docs/AZURE_CONFIG.md` - Added historical data container information

## Quick Start Guide

### Step 1: Deploy Azure Infrastructure

Deploy the updated infrastructure to create the `nbahistoricaldata` container:

```powershell
cd infra/nba
az deployment group create `
    -g nba-gbsv-model-rg `
    -f main.bicep `
    -p theOddsApiKey=... apiBasketballKey=... imageTag=<tag>
```

### Step 2: Initialize Historical Data Repository

Create the `nba-historical-data` repository:

```powershell
.\scripts\init_historical_data_repo.ps1
```

This creates the repository structure in `../nba-historical-data/` (parent directory).

### Step 3: Set Up Git Remote

```powershell
cd ../nba-historical-data
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 4: Archive Picks Before Updates

Before running fresh picks that overwrite front-end data:

```powershell
# Archive today's picks
.\scripts\archive_picks_to_azure.ps1 -Date (Get-Date -Format "yyyy-MM-dd")
```

### Step 5: Sync Historical Data

After ingesting historical data:

```powershell
# Sync to Azure
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"

# Commit to git
cd ../nba-historical-data
git add data/historical/
git commit -m "Add 2024-2025 historical data"
git push
```

## Storage Organization

### Azure Blob Storage (`nbahistoricaldata` container)

```
nbahistoricaldata/
├── archived_picks/
│   ├── {YYYY-MM-DD}/              # Picks by date
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
│       └── exports/
└── models/
    └── backtest/
        └── {version}/
            └── {YYYY-MM-DD}/
                ├── models/
                ├── results/
                └── metadata.json
```

### Local Repository (`nba-historical-data`)

```
nba-historical-data/
├── data/
│   ├── historical/                # Historical odds (committed to git)
│   ├── archived_picks/            # Archived picks (committed to git)
│   └── processed/                # Processed data
├── models/
│   ├── backtest/                 # Backtest models (committed to git)
│   └── fine_tuned/               # Fine-tuned models
└── backtest_results/             # Backtest results (committed to git)
```

## Key Features

1. **Versioning**: All data includes timestamps and model versions
2. **No Overwrites**: Unique timestamps prevent data loss
3. **Dual Storage**: Both git repository and Azure Blob Storage
4. **Organization**: Structured by date, version, and data type
5. **Metadata**: Includes version, timestamp, and source information

## Workflow Integration

### Daily Picks Workflow

```powershell
# 1. Generate picks (overwrites front-end)
.\scripts\run_slate.py --date today

# 2. Archive BEFORE overwrite
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
# 1. Ingest historical data
python scripts/ingest_historical_odds.py --season 2024-2025

# 2. Sync to Azure
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"

# 3. Commit to git
cd ../nba-historical-data
git add data/historical/
git commit -m "Add 2024-2025 historical odds"
git push
```

## Verification

### Check Azure Container

```powershell
# List containers
az storage container list --account-name nbagbsvstrg --output table

# Verify nbahistoricaldata exists
az storage container show `
    --account-name nbagbsvstrg `
    --name nbahistoricaldata
```

### List Archived Picks

```powershell
az storage blob list `
    --account-name nbagbsvstrg `
    --container-name nbahistoricaldata `
    --prefix "archived_picks/" `
    --output table
```

## Next Steps

1. **Deploy Infrastructure**: Run Bicep deployment to create the container
2. **Initialize Repository**: Run the initialization script
3. **Set Up Git Remote**: Add remote and push initial structure
4. **Test Archival**: Archive some picks to verify the workflow
5. **Sync Historical Data**: Sync existing historical data to Azure

## Related Documentation

- [Historical Data Storage Guide](./HISTORICAL_DATA_STORAGE.md) - Detailed usage guide
- [Azure Configuration](./AZURE_CONFIG.md) - Azure infrastructure details
- [Historical Data Documentation](./HISTORICAL_DATA.md) - Historical data ingestion

## Support

For issues:
1. Verify Azure CLI is installed and logged in
2. Check storage account permissions
3. Review script error messages
4. Verify file paths and permissions
