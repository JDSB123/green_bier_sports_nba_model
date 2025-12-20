# Scripts Directory

This directory contains operational scripts for the NBA v6.0 prediction system.

## Primary Scripts

### Prediction & Analysis
| Script | Description |
|--------|-------------|
| `run_slate.py` | **Main entry point** - Get predictions for today's games |
| `predict.py` | Make predictions for specific games |
| `analyze_slate_docker.py` | Analyze slate using Docker container |

### Data Collection
| Script | Description |
|--------|-------------|
| `collect_the_odds.py` | Fetch odds from The Odds API |
| `collect_api_basketball.py` | Fetch game data from API-Basketball |
| `collect_betting_splits.py` | Fetch public betting percentages |
| `collect_historical_lines.py` | Backfill historical betting lines |
| `collect_first_half_data.py` | Extract first half data from games |
| `fetch_injuries.py` | Fetch injury reports |
| `fetch_github_data.py` | Fetch backup data from GitHub |
| `fetch_external_data.py` | Fetch external data sources |
| `ingest_all.py` | Run full data ingestion pipeline |

### Training Data
| Script | Description |
|--------|-------------|
| `build_fresh_training_data.py` | **Source of truth** - Build training data from APIs |
| `build_training_dataset.py` | Build from processed CSV files |
| `build_complete_training_data.py` | Build from raw JSON files |
| `build_rich_features.py` | Build enhanced feature set |
| `generate_q1_training_data.py` | Generate Q1-specific training data |
| `generate_first_half_training_data_fast.py` | Generate 1H training data |
| `extract_betting_lines.py` | Extract consensus betting lines |

### Model Training
| Script | Description |
|--------|-------------|
| `train_models.py` | **Main trainer** - Train all 9 market models |

### Backtesting & Analysis
| Script | Description |
|--------|-------------|
| `backtest.py` | **Main backtest** - Walk-forward validation for all 9 markets |
| `analyze_backtest_results.py` | Parse and display backtest metrics |
| `analyze_roi.py` | ROI performance analysis |
| `analyze_spread_performance.py` | Spread-specific analysis |
| `calculate_pick_results.py` | Calculate pick outcomes |

### Validation
| Script | Description |
|--------|-------------|
| `validate_production_readiness.py` | Validate config, imports, API keys |
| `validate_production_current.py` | Validate current state without historical data |
| `validate_production_readiness_with_backtest.py` | Full validation + backtest |
| `validate_model.py` | Validate model files |
| `validate_leakage.py` | Check for temporal data leakage |
| `validate_card.py` | Validate betting card output |
| `verify_model_integrity.py` | Verify model checksums |
| `verify_calibration.py` | Verify probability calibration |
| `verify_container_startup.py` | Verify Docker container starts correctly |

### Operations
| Script | Description |
|--------|-------------|
| `manage_models.py` | Model file management |
| `manage_secrets.py` | Docker secrets management |
| `post_to_teams.py` | Post predictions to Microsoft Teams |
| `update_pick_tracker.py` | Update live pick tracking |
| `full_pipeline.py` | Run complete data → train → predict pipeline |

### Utilities
| Script | Description |
|--------|-------------|
| `diagnose_team_names.py` | Debug team name normalization |
| `reconcile_team_names.py` | Reconcile team name variants |
| `check_data_quality.py` | Check data quality metrics |
| `archive_processed_cache.py` | Archive old cache files |
| `test_all_api_endpoints.py` | Test all API endpoints |
| `show_executive.py` | Show executive summary |
| `review_predictions.py` | Review prediction history |
| `parse_scoreboard_results.py` | Parse scoreboard data |

### Data Processing
| Script | Description |
|--------|-------------|
| `merge_odds.py` | Merge multiple odds CSV files |
| `merge_kaggle_features.py` | Merge Kaggle features |
| `process_kaggle_stats.py` | Process Kaggle statistics |
| `process_odds_data.py` | Process raw odds data |
| `import_kaggle_betting_data.py` | Import Kaggle betting data |
| `backfill_halftime.py` | Backfill halftime data |

## Quick Start

```bash
# Get today's predictions
python scripts/run_slate.py

# Get tomorrow's predictions
python scripts/run_slate.py --date tomorrow

# Run backtest
python scripts/backtest.py

# Train all models
python scripts/train_models.py

# Validate production readiness
python scripts/validate_production_readiness.py
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
