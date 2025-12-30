# Scripts Directory

This directory contains operational scripts for the NBA v6.5 prediction system.

## Primary Scripts

### Prediction & Analysis
| Script | Description |
|--------|-------------|
| `run_slate.py` | **Main entry point** - Get predictions for today's games |
| `predict.py` | Make predictions for specific games |
| `show_executive.py` | Show executive summary |

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
| `ingest_all.py` | Run full data ingestion pipeline |

### Training Data
| Script | Description |
|--------|-------------|
| `build_fresh_training_data.py` | **Source of truth** - Build training data from APIs |
| `build_rich_features.py` | Build enhanced feature set |
| `extract_betting_lines.py` | Extract consensus betting lines |
| `validate_training_data.py` | Validate training data quality |

### Model Training
| Script | Description |
|--------|-------------|
| `train_models.py` | **Main trainer** - Train all 6 market models |
| `extract_feature_importance.py` | Extract and display feature importance |
| `log_model_performance.py` | Log model performance metrics |

### Backtesting & Analysis
| Script | Description |
|--------|-------------|
| `backtest.py` | **Main backtest** - Walk-forward validation for all 6 markets |
| `analyze_backtest_results.py` | Parse and display backtest metrics |
| `analyze_roi.py` | ROI performance analysis |
| `analyze_spread_performance.py` | Spread-specific analysis |
| `calculate_pick_results.py` | Calculate pick outcomes |

### Validation
| Script | Description |
|--------|-------------|
| `validate_production_readiness.py` | Validate config, imports, API keys |
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

### Utilities
| Script | Description |
|--------|-------------|
| `diagnose_team_names.py` | Debug team name normalization |
| `reconcile_team_names.py` | Reconcile team name variants |
| `check_data_quality.py` | Check data quality metrics |
| `test_all_api_endpoints.py` | Test all API endpoints |
| `review_predictions.py` | Review prediction history |
| `parse_scoreboard_results.py` | Parse scoreboard data |

### Data Processing
| Script | Description |
|--------|-------------|
| `process_odds_data.py` | Process raw odds data |
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
