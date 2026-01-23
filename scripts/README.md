# Scripts Directory

Operational scripts for the NBA prediction system.

Last updated: 2026-01-23

**See also:** [`docs/RUNBOOK.md`](../docs/RUNBOOK.md) for operational workflows.

---

## Canonical Training Data (Backtests)

Backtests must use the audited canonical dataset:

```
data/processed/training_data.csv
```

Notes:
- This file is the single source of truth for 2023+ odds and labels.
- Backtests do NOT rebuild or merge raw data.

Helpful commands:

```bash
# Validate canonical training data
python scripts/data_unified_validate_training.py --strict

# Download canonical training data from Azure
python scripts/download_training_data_from_azure.py --version latest --verify
```

Data engineering only (not used for backtests):

```bash
# Rebuild full training data from raw sources
python scripts/data_unified_build_training_complete.py --rebuild-from-raw
```

---

## Script Categories

### Prediction (daily use)
| Script | Description |
|--------|-------------|
| `predict_unified_slate.py` | Main entry point - get predictions for today's games |
| `predict_unified_full_game.py` | Make predictions for specific games |
| `show_executive.py` | Show executive summary |
| `predict_unified_review.py` | Review prediction results |

### Training Data
| Script | Description |
|--------|-------------|
| `data_unified_build_training_complete.py` | Data engineering only - build training data from raw sources |
| `fix_training_data_gaps.py` | Fix FG labels, totals, rest days (called by builder) |
| `data_unified_feature_complete.py` | Compute model features (called by builder) |
| `data_unified_validate_training.py` | Validate training data quality |
| `data_unified_compute_betting_labels.py` | Compute spread/total labels |
| `data_unified_build_training_fresh.py` | Verify/copy canonical training data (no rebuilds) |

### Model Training & Backtesting
| Script | Description |
|--------|-------------|
| `model_train_all.py` | Main trainer - train all market models |
| `historical_backtest_production.py` | Backtest production models |
| `optimize_confidence_thresholds.py` | Sweep confidence/edge thresholds for best ROI/accuracy |

### Data Ingestion
| Script | Description |
|--------|-------------|
| `data_unified_ingest_all.py` | Run full ingestion pipeline |
| `data_unified_ingest_database.py` | Ingest wyattowalsh/basketball dataset |
| `historical_ingest_elo_ratings.py` | Historical: ingest FiveThirtyEight ELO ratings |
| `historical_ingest_period_odds.py` | Historical: ingest period odds from The Odds API |
| `data_unified_fetch_the_odds.py` | Fetch current odds from The Odds API |
| `data_unified_fetch_api_basketball.py` | Fetch game data from API-Basketball |
| `data_unified_fetch_betting_splits.py` | Fetch public betting percentages |
| `data_unified_fetch_injuries.py` | Fetch injury reports |
| `historical_fetch_quarter_scores.py` | Historical: fetch quarter-by-quarter scores |
| `data_unified_fetch_box_scores.py` | Fetch NBA API box scores |
| `download_kaggle_player_data.py` | Download Kaggle NBA player box scores |
| `data_unified_extract_betting_lines.py` | Extract betting lines from odds data |

### Azure Blob Storage
| Script | Description |
|--------|-------------|
| `upload_training_data_to_azure.py` | Validate then upload to Azure |
| `download_training_data_from_azure.py` | Download canonical training data from Azure |

### Validation
| Script | Description |
|--------|-------------|
| `predict_validate_production_readiness.py` | Validate config, imports, API keys |
| `model_validate.py` | Validate model files |
| `data_unified_validate_training.py` | Validate training data |
| `ci_sanity_check.py` | CI/CD sanity checks |
| `predict_test_all_api_endpoints.py` | Test all API endpoints |

### Analysis & Export
| Script | Description |
|--------|-------------|
| `calculate_pick_results.py` | Calculate pick outcomes |
| `export_executive_html.py` | Export executive HTML summary |
| `historical_export_odds.py` | Export historical odds |
| `historical_export_period_odds.py` | Export period odds to CSV |
| `update_pick_tracker.py` | Update pick tracking database |

### Operations
| Script | Description |
|--------|-------------|
| `model_manage.py` | Model file management |
| `manage_secrets.py` | Docker secrets management |
| `post_to_teams.py` | Post predictions to Microsoft Teams |
| `prepare_deployment.py` | Prepare deployment package |
| `bump_version.py` | Bump version number |

### PowerShell/Shell
| Script | Description |
|--------|-------------|
| `deploy.ps1` | Deploy to Azure |
| `deploy_production.ps1` | Production deployment |
| `archive_picks_to_azure.ps1` | Archive picks to Azure Blob |
| `sync_archives_to_azure.ps1` | Sync archives to Azure |
| `sync_historical_data_to_azure.ps1` | Sync historical data |
| `cleanup_nba_docker.ps1` | Clean up Docker resources |
| `setup_codespace.sh` | One-command dev setup (env, deps, hooks) |
| `watch_actions.sh` | Watch latest GitHub Actions run for a branch |

---

## Data Coverage (as of 2026-01-11)

| Data Type | Coverage | Notes |
|-----------|----------|-------|
| Training Data | 2023-01-01 to 2026-01-09 | 3,969 games, 327 columns |
| FG Labels | 100% | fg_spread_covered, fg_total_over, fg_home_win |
| 1H Labels | 100% | 1h_spread_covered, 1h_total_over |
| Model Features | 55/55 (100%) | All required features present |
| Odds Coverage | FG 100% / 1H 78.4% | Full-game lines complete; 1H partial |
| Injury Impact | 100% | Via Kaggle box score inference |

---

## Azure Blob Storage (Single Source of Truth)

Training data is stored in Azure as the single source of truth:

```
Storage Account: nbagbsvstrg
Container: nbahistoricaldata
Prefix: training_data/

training_data/
  v2026.01.11/                    # Versioned release
    training_data.csv
    manifest.json
  latest/                         # Always points to validated version
    training_data.csv
    manifest.json
```

---

## Quick Start

```bash
# Get today's predictions
python scripts/predict_unified_slate.py

# Validate canonical training data
python scripts/data_unified_validate_training.py --strict

# Train models
python scripts/model_train_all.py

# Run backtest
python scripts/historical_backtest_production.py --data data/processed/training_data.csv
```

## Docker Usage

Most scripts are designed to run locally and orchestrate Docker containers.
The prediction API runs inside Docker on port 8090.

```bash
# Start Docker stack
docker compose up -d

# Run predictions via Docker
python scripts/predict_unified_slate.py
```
