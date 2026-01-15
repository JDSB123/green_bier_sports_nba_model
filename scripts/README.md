# Scripts Directory

Operational scripts for the NBA prediction system.

Script count: 40 Python scripts + 11 PowerShell/Shell scripts
Last updated: 2026-01-11

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
python scripts/validate_training_data.py --strict

# Download canonical training data from Azure
python scripts/download_training_data_from_azure.py --version latest --verify
```

Data engineering only (not used for backtests):

```bash
# Rebuild full training data from raw sources
python scripts/build_training_data_complete.py
```

---

## Script Categories

### Prediction (daily use)
| Script | Description |
|--------|-------------|
| `run_slate.py` | Main entry point - get predictions for today's games |
| `predict.py` | Make predictions for specific games |
| `show_executive.py` | Show executive summary |
| `review_predictions.py` | Review prediction results |

### Training Data
| Script | Description |
|--------|-------------|
| `build_training_data_complete.py` | Data engineering only - build training data from raw sources |
| `fix_training_data_gaps.py` | Fix FG labels, totals, rest days (called by builder) |
| `complete_training_features.py` | Compute model features (called by builder) |
| `validate_training_data.py` | Validate training data quality |
| `compute_betting_labels.py` | Compute spread/total labels |
| `build_fresh_training_data.py` | Verify/copy canonical training data (no rebuilds) |

### Model Training & Backtesting
| Script | Description |
|--------|-------------|
| `train_models.py` | Main trainer - train all market models |
| `backtest_production.py` | Backtest production models |
| `optimize_confidence_thresholds.py` | Sweep confidence/edge thresholds for best ROI/accuracy |

### Data Ingestion
| Script | Description |
|--------|-------------|
| `ingest_all.py` | Run full ingestion pipeline |
| `ingest_nba_database.py` | Ingest wyattowalsh/basketball dataset |
| `ingest_elo_ratings.py` | Ingest FiveThirtyEight ELO ratings |
| `ingest_historical_period_odds.py` | Ingest period odds from The Odds API |
| `collect_the_odds.py` | Fetch current odds from The Odds API |
| `collect_api_basketball.py` | Fetch game data from API-Basketball |
| `collect_betting_splits.py` | Fetch public betting percentages |
| `fetch_injuries.py` | Fetch injury reports |
| `fetch_quarter_scores.py` | Fetch quarter-by-quarter scores |
| `fetch_nba_box_scores.py` | Fetch NBA API box scores |
| `download_kaggle_player_data.py` | Download Kaggle NBA player box scores |
| `extract_betting_lines.py` | Extract betting lines from odds data |

### Azure Blob Storage
| Script | Description |
|--------|-------------|
| `upload_training_data_to_azure.py` | Validate then upload to Azure |
| `download_training_data_from_azure.py` | Download canonical training data from Azure |

### Validation
| Script | Description |
|--------|-------------|
| `validate_production_readiness.py` | Validate config, imports, API keys |
| `validate_model.py` | Validate model files |
| `validate_training_data.py` | Validate training data |
| `ci_sanity_check.py` | CI/CD sanity checks |
| `test_all_api_endpoints.py` | Test all API endpoints |

### Analysis & Export
| Script | Description |
|--------|-------------|
| `calculate_pick_results.py` | Calculate pick outcomes |
| `export_executive_html.py` | Export executive HTML summary |
| `export_historical_odds.py` | Export historical odds |
| `export_period_odds_to_csv.py` | Export period odds to CSV |
| `update_pick_tracker.py` | Update pick tracking database |

### Operations
| Script | Description |
|--------|-------------|
| `manage_models.py` | Model file management |
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
python scripts/run_slate.py

# Validate canonical training data
python scripts/validate_training_data.py --strict

# Train models
python scripts/train_models.py

# Run backtest
python scripts/backtest_production.py --data data/processed/training_data.csv
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
