# Scripts Directory

**Last Cleanup:** 2026-01-23 | **Scripts:** 18 (down from 73)

This directory contains ONLY the essential scripts needed for production workflows.
One-time utilities and deprecated scripts are in `archive/legacy_scripts/` and `archive/legacy_data_scripts/`.

---

## Production Workflows

### 🎯 Daily Predictions (VS Code Tasks)

| Script | Purpose | VS Code Task |
|--------|---------|--------------|
| `predict_unified_full_game.py` | Generate predictions for today | "Generate Predictions" |
| `predict_preflight_freshness.py` | Pre-prediction validation | "Preflight: Freshness & Invariants" |
| `data_unified_fetch_the_odds.py` | Fetch live odds | "Collect Odds Data" |

### 🏋️ Model Training

| Script | Purpose |
|--------|---------|
| `model_train_all.py` | Train all 4 models (1h_spread, 1h_total, fg_spread, fg_total) |
| `data_unified_build_training_complete.py` | Build training data from raw sources |
| `data_unified_feature_complete.py` | Feature engineering (called by build_training) |

### 📊 Backtesting

| Script | Purpose |
|--------|---------|
| `historical_backtest_production.py` | Production backtesting |
| `historical_backtest_extended.py` | Extended historical backtest |

### ✅ Validation & CI

| Script | Purpose |
|--------|---------|
| `validate_environment.py` | Validate Python environment |
| `data_unified_validate_training.py` | Validate training data |
| `predict_validate_production_readiness.py` | Production readiness check |
| `predict_test_all_api_endpoints.py` | Test API endpoints |
| `ci_sanity_check.py` | CI validation |

### 🔧 Utilities

| Script | Purpose |
|--------|---------|
| `bump_version.py` | Bump VERSION file |
| `post_to_teams.py` | Post picks to MS Teams |
| `download_training_data_from_azure.py` | Download training data from Azure |
| `upload_training_data_to_azure.py` | Upload training data to Azure |
| `data_unified_ingest_all.py` | Bulk data ingestion (rarely needed) |

---

## Script Entry Points Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY PREDICTION FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│  1. predict_preflight_freshness.py  → Validate environment      │
│  2. data_unified_fetch_the_odds.py  → Get live betting lines    │
│  3. predict_unified_full_game.py    → Generate predictions      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│  1. data_unified_build_training_complete.py                     │
│       └─ calls data_unified_feature_complete.py                 │
│       └─ outputs: data/processed/training_data.csv              │
│  2. model_train_all.py                                          │
│       └─ outputs: models/production/*.joblib                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Archived Scripts

Scripts moved to `archive/` during 2026-01-23 cleanup:

- **archive/legacy_data_scripts/** (14 scripts) - One-time data builds, audits
- **archive/legacy_scripts/** (12 scripts) - Deprecated prediction/utility scripts

These can be recovered if needed but should not be used for production workflows.
