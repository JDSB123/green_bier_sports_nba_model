# Historical Data Storage & Maintenance Best Practices

## Overview

This document outlines best practices for managing historical data across the NBA prediction system's three-tier architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LOCAL (nba_main/)                                                      │
│   ├── Development, backtesting, model training                          │
│   ├── Historical data ingestion                                         │
│   └── Model artifact creation                                           │
│            │                                                             │
│            │ git push                                                    │
│            ▼                                                             │
│   GIT REPOSITORY                                                         │
│   ├── Version control for code + critical data                          │
│   ├── Protected historical datasets                                     │
│   └── Production model artifacts                                        │
│            │                                                             │
│            │ CI/CD deploy                                                │
│            ▼                                                             │
│   AZURE (nba-gbsv-model-rg)                                             │
│   ├── Container App: Live predictions API                               │
│   ├── Blob Storage: Large data + archived picks                         │
│   └── Key Vault: Secrets                                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Two-Phase Architecture

### Phase 1: Backtesting (Local + Git)

**Purpose:** Validate model accuracy using historical data

**Data Location:** Local filesystem, committed to Git

```
data/historical/           # COMMITTED TO GIT
├── the_odds/              # Raw historical JSON (immutable)
│   ├── events/            # Game events by season
│   ├── odds/              # Full game odds
│   ├── period_odds/       # First half odds
│   └── metadata/          # Ingestion progress
├── exports/               # Processed CSV/Parquet
│   ├── {season}_odds_featured.csv
│   ├── {season}_odds_1h.csv
│   └── {season}_events.csv
├── derived/               # Consensus lines
│   └── theodds_lines.csv
└── elo/                   # ELO ratings
    └── fivethirtyeight_elo_historical.csv

data/external/             # COMMITTED TO GIT
└── kaggle/
    └── nba_2008-2025.csv  # 17 seasons game data
```

**Why Git?**
- Reproducibility: Anyone can clone and run backtests
- Immutability: Historical data doesn't change
- Version control: Track data updates with commits
- Size: ~100MB compressed (acceptable for Git)

### Phase 2: Live Predictions (Azure)

**Purpose:** Serve real-time predictions via API

**Data Location:** Azure Container App + Blob Storage

```
Azure Container App (nba-gbsv-api)
├── models/production/     # Baked into Docker image
│   ├── fg_spread_model.joblib
│   ├── fg_total_model.joblib
│   ├── 1h_spread_model.joblib
│   ├── 1h_total_model.joblib
│   └── model_pack.json

Azure Blob Storage (nbagbsvstrg)
├── models/                # Model artifact backup
├── predictions/           # Live prediction outputs
├── results/               # Graded picks with outcomes
└── nbahistoricaldata/     # Large historical data backup
    ├── archived_picks/    # Daily picks with timestamps
    ├── historical/        # Blob backup of historical data
    └── models/backtest/   # Backtest model artifacts
```

---

## Data Classification

### Tier 1: Committed to Git (Protected)

| Data | Location | Reason |
|------|----------|--------|
| Historical odds (JSON) | `data/historical/the_odds/` | Reproducibility |
| Historical exports (CSV) | `data/historical/exports/` | Analysis-ready |
| Derived lines | `data/historical/derived/` | Backtest input |
| Kaggle dataset | `data/external/kaggle/` | Long-term history |
| ELO ratings | `data/historical/elo/` | Feature data |
| Production models | `models/production/` | Deployment artifacts |
| Archive (picks audit) | `archive/` | ROI verification |

### Tier 2: Git Ignored (Regenerated)

| Data | Location | Reason |
|------|----------|--------|
| Raw API responses | `data/raw/` | Temporary, regenerated |
| Processed training data | `data/processed/` | Derived from raw |
| Logs | `logs/` | Ephemeral |
| Temp outputs | `nba_picks_*.html` | Generated per run |

### Tier 3: Azure Blob Storage (Backup + Overflow)

| Data | Container | Reason |
|------|-----------|--------|
| Archived picks | `nbahistoricaldata` | Audit trail, large volume |
| Historical backup | `nbahistoricaldata` | Disaster recovery |
| Backtest models | `nbahistoricaldata` | Version comparison |
| Prediction outputs | `predictions` | Live API results |

---

## Workflow Best Practices

### 1. Historical Data Ingestion

```powershell
# Step 1: Ingest on LOCAL
python scripts/ingest_historical_odds.py --season 2024-2025

# Step 2: Export to analysis-ready format
python scripts/export_historical_odds.py
python scripts/export_period_odds_to_csv.py

# Step 3: Rebuild derived lines
python scripts/rebuild_derived_lines.py

# Step 4: Commit to Git (protect the data)
git add data/historical/
git commit -m "Add 2024-2025 historical odds data"
git push

# Step 5: Backup to Azure (optional, for DR)
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"
```

### 2. Model Training & Backtesting

```powershell
# Step 1: Build training data from historical
python scripts/build_complete_training_data.py

# Step 2: Train models
python scripts/train_production_models.py

# Step 3: Run backtest
python scripts/backtest_production_model.py \
    --start-date 2024-10-01 \
    --end-date 2025-01-01

# Step 4: Commit validated models
git add models/production/
git commit -m "Update production models - NBA_v33.0.12.0"
git push
```

### 3. Deploying to Production

```powershell
# Step 1: Tag the release
git tag -a NBA_v33.0.12.0 -m "Production release with 1H moneyline"
git push --tags

# Step 2: Build and push Docker image
$VERSION = Get-Content VERSION
docker build -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined .
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Step 3: Deploy to Container App
pwsh ./infra/nba/deploy.ps1 -Tag $VERSION

# Step 4: Archive current predictions (before overwrite)
.\scripts\archive_picks_to_azure.ps1 -Date (Get-Date -Format "yyyy-MM-dd")
```

---

## Data Integrity Rules

### Rule 1: Historical Data is Immutable
```
Once ingested and committed:
- DO NOT modify historical odds files
- DO NOT delete historical events
- DO append new seasons as they complete

If corrections needed:
- Create new files with "_corrected" suffix
- Document the correction in CHANGELOG.md
```

### Rule 2: Derived Data Can Be Regenerated
```
Files that CAN be deleted and rebuilt:
- data/historical/exports/*.csv
- data/historical/derived/*.csv
- data/processed/*.csv

Because they are derived from:
- data/historical/the_odds/*.json (raw, immutable)
- data/external/kaggle/*.csv (static snapshot)
```

### Rule 3: Production Models Are Tagged
```
Every production model must have:
1. model_pack.json with version, date, backtest results
2. Git tag matching the version
3. Docker image tagged with the version

Example:
- models/production/model_pack.json → version: "NBA_v33.0.12.0"
- git tag NBA_v33.0.12.0
- nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.12.0
```

### Rule 4: Separate Backtest from Live Data
```
BACKTEST data:
- data/historical/ (committed, static)
- data/external/kaggle/ (committed, static)
- data/backtest_results/ (committed, results)

LIVE data:
- data/raw/ (ignored, ephemeral)
- data/processed/ (ignored, regenerated)
- Azure Blob predictions/ (archived)
```

---

## Storage Size Management

### Current Repository Size

| Path | Size | In Git? |
|------|------|---------|
| `data/historical/the_odds/` | ~50 MB | YES |
| `data/historical/exports/` | ~30 MB | YES |
| `data/external/kaggle/` | ~5 MB | YES |
| `models/production/` | ~10 MB | YES |
| **Total Committed** | ~100 MB | - |

### Size Limits

| Tier | Limit | Action if Exceeded |
|------|-------|-------------------|
| Git repo | 500 MB | Move large files to Azure Blob |
| Git file | 100 MB | Use Git LFS or Azure Blob |
| Azure Blob | No limit | Archive old data |

### If Repository Gets Too Large

```powershell
# Option 1: Use Git LFS for large files
git lfs install
git lfs track "data/historical/exports/*.parquet"

# Option 2: Move to Azure Blob only
.\scripts\sync_historical_data_to_azure.ps1
# Then add to .gitignore

# Option 3: Separate historical data repo
# (Already planned: nba-historical-data repo)
```

---

## Backup & Disaster Recovery

### Git Repository (Primary)
```
- Hosted on GitHub/Azure DevOps
- All historical data committed
- All production models committed
- Clone to restore: git clone <repo-url>
```

### Azure Blob Storage (Backup)
```
Container: nbahistoricaldata
- Synced from local: .\scripts\sync_historical_data_to_azure.ps1
- Archived picks: .\scripts\archive_picks_to_azure.ps1
- Restore: az storage blob download-batch ...
```

### Recovery Scenarios

| Scenario | Recovery |
|----------|----------|
| Local data lost | `git clone` |
| Git repo deleted | Restore from Azure Blob |
| Azure region down | Git repo is source of truth |
| Model corruption | `git checkout` previous tag |

---

## Checklist: Before Each Season

1. [ ] Ingest completed season's historical data
2. [ ] Export to CSV/Parquet formats
3. [ ] Rebuild derived lines with new data
4. [ ] Commit to Git with descriptive message
5. [ ] Sync backup to Azure Blob
6. [ ] Update model_pack.json with new date range
7. [ ] Run full backtest on expanded dataset
8. [ ] Retrain models if performance improved
9. [ ] Tag release with new version
10. [ ] Deploy updated container

---

## Quick Reference

### Key Directories

| Purpose | Path |
|---------|------|
| Raw historical (immutable) | `data/historical/the_odds/` |
| Exported historical (derived) | `data/historical/exports/` |
| Derived lines (backtest input) | `data/historical/derived/` |
| External datasets | `data/external/` |
| Production models | `models/production/` |
| Backtest results | `data/backtest_results/` |

### Key Scripts

| Purpose | Script |
|---------|--------|
| Ingest historical odds | `scripts/ingest_historical_odds.py` |
| Export to CSV | `scripts/export_historical_odds.py` |
| Export 1H to CSV | `scripts/export_period_odds_to_csv.py` |
| Rebuild derived lines | `scripts/rebuild_derived_lines.py` |
| Train models | `scripts/train_production_models.py` |
| Run backtest | `scripts/backtest_production_model.py` |
| Sync to Azure | `scripts/sync_historical_data_to_azure.ps1` |
| Archive picks | `scripts/archive_picks_to_azure.ps1` |

### Key Azure Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Container App | `nba-gbsv-api` | Live predictions API |
| Storage Account | `nbagbsvstrg` | Blob storage |
| Blob Container | `nbahistoricaldata` | Historical backup |
| Container Registry | `nbagbsacr` | Docker images |
| Key Vault | `nbagbs-keyvault` | API secrets |
