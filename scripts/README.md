# Scripts Directory

Operational scripts for the NBA prediction system.

**Script Count:** 39 Python scripts + 11 PowerShell/Shell scripts  
**Last Updated:** 2026-01-11

---

## ⚠️ SINGLE SOURCE OF TRUTH

**Training Data Pipeline:** One master script builds ALL training data.

```bash
# BUILD TRAINING DATA (the ONLY way to build training data)
python scripts/build_training_data_complete.py --start-date 2023-01-01

# This script:
# 1. Merges ALL data sources (Kaggle, TheOdds, nba_api, etc.)
# 2. Automatically calls fix_training_data_gaps.py
# 3. Automatically calls complete_training_features.py
# 4. Outputs (canonical): data/processed/training_data.csv
# 5. Outputs (snapshot): data/processed/training_data_complete_<YYYY>.csv
```

---

## Script Categories

### 🎯 PREDICTION (Daily Use)
| Script | Description |
|--------|-------------|
| `run_slate.py` | **Main entry point** - Get predictions for today's games |
| `predict.py` | Make predictions for specific games |
| `show_executive.py` | Show executive summary |
| `review_predictions.py` | Review prediction results |

### 📦 TRAINING DATA
| Script | Description |
|--------|-------------|
| `build_training_data_complete.py` | **MASTER BUILDER** - Build ALL training data from ALL sources |
| `fix_training_data_gaps.py` | Fix FG labels, totals, rest days (called by master) |
| `complete_training_features.py` | Compute all model features (called by master) |
| `validate_training_data.py` | Validate training data quality |
| `compute_betting_labels.py` | Compute spread_covered, total_over labels |

### 🔧 MODEL TRAINING & BACKTESTING
| Script | Description |
|--------|-------------|
| `train_models.py` | **Main trainer** - Train all market models |
| `backtest_production.py` | Backtest production models |

### 📥 DATA INGESTION
| Script | Description |
|--------|-------------|
| `ingest_all.py` | Run full ingestion pipeline |
| `ingest_nba_database.py` | Ingest wyattowalsh/basketball dataset |
| `ingest_elo_ratings.py` | Ingest FiveThirtyEight ELO ratings |
| `ingest_historical_period_odds.py` | Ingest period odds from TheOdds |
| `collect_the_odds.py` | Fetch current odds from The Odds API |
| `collect_api_basketball.py` | Fetch game data from API-Basketball |
| `collect_betting_splits.py` | Fetch public betting percentages |
| `fetch_injuries.py` | Fetch injury reports |
| `fetch_quarter_scores.py` | Fetch quarter-by-quarter scores |
| `fetch_nba_box_scores.py` | Fetch NBA API box scores |
| `download_kaggle_player_data.py` | Download Kaggle NBA player box scores |
| `extract_betting_lines.py` | Extract betting lines from odds data |

### ☁️ AZURE BLOB STORAGE
| Script | Description |
|--------|-------------|
| `upload_training_data_to_azure.py` | **Quality-gate upload** - Validates then uploads to Azure |
| `download_training_data_from_azure.py` | Download canonical training data from Azure |

### ✅ VALIDATION
| Script | Description |
|--------|-------------|
| `validate_production_readiness.py` | Validate config, imports, API keys |
| `validate_model.py` | Validate model files |
| `validate_training_data.py` | Validate training data |
| `ci_sanity_check.py` | CI/CD sanity checks |
| `test_all_api_endpoints.py` | Test all API endpoints |

### 📊 ANALYSIS & EXPORT
| Script | Description |
|--------|-------------|
| `calculate_pick_results.py` | Calculate pick outcomes |
| `export_executive_html.py` | Export executive HTML summary |
| `export_historical_odds.py` | Export historical odds |
| `export_period_odds_to_csv.py` | Export period odds to CSV |
| `update_pick_tracker.py` | Update pick tracking database |

### 🔧 OPERATIONS
| Script | Description |
|--------|-------------|
| `manage_models.py` | Model file management |
| `manage_secrets.py` | Docker secrets management |
| `post_to_teams.py` | Post predictions to Microsoft Teams |
| `prepare_deployment.py` | Prepare deployment package |
| `bump_version.py` | Bump version number |

### 📜 POWERSHELL/SHELL SCRIPTS
| Script | Description |
|--------|-------------|
| `deploy.ps1` | Deploy to Azure |
| `deploy_production.ps1` | Production deployment |
| `archive_picks_to_azure.ps1` | Archive picks to Azure Blob |
| `sync_archives_to_azure.ps1` | Sync archives to Azure |
| `sync_historical_data_to_azure.ps1` | Sync historical data |
| `cleanup_nba_docker.ps1` | Clean up Docker resources |

---

## Data Coverage (as of 2026-01-11)

| Data Type | Coverage | Notes |
|-----------|----------|-------|
| **Training Data** | 2023-01-01 to 2026-01-09 | 3,969 games, 327 columns |
| **FG Labels** | 100% | spread_covered, total_over, home_win |
| **1H Labels** | 100% | 1h_spread_covered, 1h_total_over |
| **Model Features** | 55/55 (100%) | All required features present |
| **Odds Coverage** | 100% | Spread + total for all games |
| **Injury Impact** | 100% | Via Kaggle box score inference |

---

## Azure Blob Storage (Single Source of Truth)

Training data is stored in Azure as the single source of truth:
```
Storage Account: nbagbsvstrg
Container: nbahistoricaldata
Prefix: training_data/

training_data/
├── v2026.01.11/                    # Versioned release
│   ├── training_data_complete_2023_with_injuries.csv
│   └── manifest.json
└── latest/                         # Always points to validated version
    ├── training_data_complete_2023_with_injuries.csv
    └── manifest.json
```

---

## Quick Start

```bash
# Get today's predictions
python scripts/run_slate.py

# Build training data from scratch
python scripts/build_training_data_complete.py

# Train models
python scripts/train_models.py

# Run backtest
python scripts/backtest_production.py

# Upload validated training data to Azure
python scripts/upload_training_data_to_azure.py
```

## Docker Usage

Most scripts are designed to run locally and orchestrate Docker containers.
The prediction API runs inside Docker on port 8090.

```bash
# Start Docker stack
docker compose up -d

# Run predictions via Docker
python scripts/run_slate.py
```
