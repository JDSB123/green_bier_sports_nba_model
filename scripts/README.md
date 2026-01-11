# Scripts Directory

Operational scripts for the NBA prediction system.

## ⚠️ SINGLE SOURCE OF TRUTH

**Training Data Pipeline:** One master script builds ALL training data.

```bash
# BUILD TRAINING DATA (the ONLY way to build training data)
python scripts/build_training_data_complete.py --start-date 2023-01-01

# This script:
# 1. Merges ALL data sources (Kaggle, TheOdds, nba_api, etc.)
# 2. Automatically calls fix_training_data_gaps.py
# 3. Automatically calls complete_training_features.py
# 4. Outputs: data/processed/training_data_complete_2023.csv (55 features, 100% coverage)
```

---

## Script Categories

### 🎯 PREDICTION (Daily Use)
| Script | Description |
|--------|-------------|
| `run_slate.py` | **Main entry point** - Get predictions for today's games |
| `predict.py` | Make predictions for specific games |
| `show_executive.py` | Show executive summary |

### 📦 TRAINING DATA (Single Source of Truth)
| Script | Description |
|--------|-------------|
| `build_training_data_complete.py` | **MASTER BUILDER** - Build ALL training data from ALL sources. Calls gap fixes automatically. |
| `fix_training_data_gaps.py` | Fix FG labels, totals, rest days (called by master) |
| `complete_training_features.py` | Compute all 55 model features (called by master) |
| `validate_training_data.py` | Validate training data quality |

### 🔧 MODEL TRAINING
| Script | Description |
|--------|-------------|
| `train_models.py` | **Main trainer** - Train all 4 market models |
| `backtest_production.py` | Backtest production models |

### 📥 DATA INGESTION
| Script | Description |
|--------|-------------|
| `ingest_all.py` | Run full ingestion pipeline |
| `ingest_nba_database.py` | Ingest wyattowalsh/basketball (65K games, Q1-Q4, play-by-play) |
| `ingest_elo_ratings.py` | Ingest FiveThirtyEight ELO ratings |
| `ingest_historical_period_odds.py` | Ingest period odds from TheOdds |
| `collect_the_odds.py` | Fetch current odds from The Odds API |
| `collect_api_basketball.py` | Fetch game data from API-Basketball |
| `collect_betting_splits.py` | Fetch public betting percentages |
| `collect_first_half_data.py` | Extract first half data |
| `fetch_injuries.py` | Fetch injury reports |
| `fetch_quarter_scores.py` | Fetch quarter-by-quarter scores |
| `fetch_box_scores_parallel.py` | Fetch box scores in parallel |
| `fetch_nba_box_scores.py` | Fetch NBA API box scores |
| `rebuild_derived_lines.py` | Rebuild TheOdds derived lines CSV |

### ✅ VALIDATION
| Script | Description |
|--------|-------------|
| `validate_production_readiness.py` | Validate config, imports, API keys |
| `validate_model.py` | Validate model files |
| `validate_leakage.py` | Check for temporal data leakage |
| `validate_card.py` | Validate betting card output |
| `validate_training_data.py` | Validate training data |
| `verify_model_integrity.py` | Verify model checksums |
| `verify_calibration.py` | Verify probability calibration |

### 📊 ANALYSIS & EXPORT
| Script | Description |
|--------|-------------|
| `calculate_pick_results.py` | Calculate pick outcomes |
| `export_card_html.py` | Export HTML betting card |
| `export_comprehensive_html.py` | Export comprehensive HTML report |
| `export_executive_html.py` | Export executive HTML summary |
| `export_table_html.py` | Export HTML tables |
| `export_historical_odds.py` | Export historical odds |
| `export_period_odds_to_csv.py` | Export period odds to CSV |

### 🔧 OPERATIONS
| Script | Description |
|--------|-------------|
| `manage_models.py` | Model file management |
| `manage_secrets.py` | Docker secrets management |
| `post_to_teams.py` | Post predictions to Microsoft Teams |
| `deploy.ps1` | Deploy to production |

---

## Data Coverage (as of 2026-01-11)

| Data Type | Coverage | Notes |
|-----------|----------|-------|
| **Training Data** | 2023-01-01 to 2026-01-09 | 3,969 games, 324 columns |
| **FG Labels** | 100% | spread_covered, total_over, home_win |
| **1H Labels** | 100% | 1h_spread_covered, 1h_total_over |
| **Model Features** | 55/55 (100%) | All required features present |
| **Moneylines** | 69.5% | Best available from TheOdds |
| **Injury Impact** | Baseline only | inactive_players.csv exists but no PPG data |

## Archive

Deprecated scripts moved to `scripts/archive/`:
- `prepare_kaggle_training_data.py` - Superseded by build_training_data_complete.py

---

## Quick Start

```bash
# Get today's predictions
python scripts/run_slate.py

# Get tomorrow's predictions
python scripts/run_slate.py --date tomorrow

# Run backtest
python scripts/backtest.py

# Backtest the actual production artifacts (recommended for release validation)
python scripts/backtest_production_model.py --data data/processed/training_data_theodds.csv --models-dir models/production --markets all

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
