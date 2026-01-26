# Scripts Directory

**Last Cleanup:** 2026-01-26 | **Scripts:** 18

This directory contains ONLY the essential scripts needed for production workflows.

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

### ✅ Validation & CI

| Script | Purpose |
|--------|---------|
| `validate_environment.py` | Validate Python environment |
| `data_unified_validate_training.py` | Validate training data |
| `predict_validate_production_readiness.py` | Production readiness check (`--live` runs live endpoints + end-to-end pipeline) |
| `check_production_runtime_isolation.py` | Production runtime isolation test (no coverage threshold) |
| `predict_test_all_api_endpoints.py` | Test API endpoints |
| `ci_sanity_check.py` | CI validation |

### 🔧 Utilities

| Script | Purpose |
|--------|---------|
| `bump_version.py` | Bump VERSION file |
| `post_to_teams.py` | Post picks to MS Teams |
| `post_to_teams_scheduled.py` | Scheduled Teams poster (uses API slate window + hourly cadence) |
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
