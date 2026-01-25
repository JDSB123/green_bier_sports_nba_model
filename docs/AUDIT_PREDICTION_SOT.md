# Prediction Model Single Source of Truth Audit (NBA)

## Scope
Prediction models only (1H/FG spreads & totals). Excludes historical/backtest workflows unless explicitly referenced.

## 1) Feature schema vs runtime payload
- **Training schema**: `UNIFIED_FEATURE_NAMES` (102 features) in [src/modeling/unified_features.py](../src/modeling/unified_features.py).
- **Prediction contract**: `models/production/model_features.json` (source of truth for required features per market).
- **Runtime payload**: `RichFeatureBuilder.build_game_features()` in [src/features/rich_features.py](../src/features/rich_features.py).

### Current state (fixed)
- Prediction-time contract is generated directly from production `.joblib` artifacts.
- Startup checks and production readiness validation now read the contract.
- RichFeatureBuilder can skip expensive computations when contract is provided.

**Conclusion:** training schema ≠ runtime payload, but prediction-time requirements are now explicit and enforced.

## 2) Feature integrity checks
Startup integrity validates:
- Required markets (from `model_pack.json`)
- Feature availability (from `model_features.json` when present)
- RichFeatureBuilder key coverage (static scan for safety)

## 3) Canonicalization/standardization
- **ESPN-format names**: `src/ingestion/standardize.py`
- **Canonical IDs & variants**: `src/utils/team_names.py`

Both share the same `team_mapping.json`, but serve different layers (ingestion vs canonical IDs).

## 4) Bicep entry points (prediction vs historical)
Single entry point currently:
- [infra/nba/main.bicep](../infra/nba/main.bicep)

This file currently mixes platform + compute + optional components (no explicit prediction-only module boundary).

---

# Recommended Fix Plan (Implementation Pending)

## A) Define the actual prediction feature contract
1. Create a **single, generated feature contract** from model artifacts:
   - Source: `feature_columns` in `models/production/*.joblib`
   - Output: `models/production/model_features.json` (single authoritative contract)
2. Enforce the contract in `startup_checks` and `feature_validation`.

## B) Separate schema vs payload
- Rename or clarify:
  - `unified_features.py` → **training_schema.py** (or equivalent)
  - `RichFeatureBuilder` → **PredictionFeatureBuilder**

## C) Standardize canonicalization
- Select **one canonical module** as the single entry point for team normalization.
- Make others thin wrappers or deprecate entirely.

## D) Bicep single entry point for prediction only
- Added a dedicated prediction-only entry point:
  - `infra/nba/prediction.bicep`
- `infra/nba/main.bicep` remains the full-stack entry point (Teams Bot optional).

---

# Script Naming Cleanup (Proposed Mapping)
This is a proposed mapping; needs approval before renames.

## Data ingestion / unified datasets
- `ingest_all.py` → `data_unified_ingest_all.py`
- `ingest_nba_database.py` → `data_unified_ingest_database.py`
- `collect_api_basketball.py` → `data_unified_fetch_api_basketball.py`
- `collect_the_odds.py` → `data_unified_fetch_the_odds.py`
- `collect_betting_splits.py` → `data_unified_fetch_betting_splits.py`
- `fetch_nba_box_scores.py` → `data_unified_fetch_box_scores.py`
- `fetch_injuries.py` → `data_unified_fetch_injuries.py`

## Prediction
- `predict.py` → `predict_unified_full_game.py`
- `run_slate.py` → `predict_unified_slate.py`
- `save_daily_picks.py` → `predict_unified_save_daily_picks.py`
- `review_predictions.py` → `predict_unified_review.py`

## Training / modeling
- `train_models.py` → `model_train_all.py`
- `build_training_data_complete.py` → `data_unified_build_training_complete.py`
- `build_fresh_training_data.py` → `data_unified_build_training_fresh.py`
- `create_master_training_data.py` → `data_unified_build_master_training.py`
- `complete_training_features.py` → `data_unified_feature_complete.py`
- `compute_betting_labels.py` → `data_unified_compute_betting_labels.py`
- `update_training_betting_lines.py` → `data_unified_update_training_lines.py`
- `validate_training_data.py` → `data_unified_validate_training.py`

## Backtest / historical
- `backtest_production.py` → `historical_backtest_production.py`
- `backtest_extended.py` → `historical_backtest_extended.py`
- `export_historical_odds.py` → `historical_export_odds.py`
- `export_period_odds_to_csv.py` → `historical_export_period_odds.py`
- `audit_historical_data_integrity.py` → `historical_audit_data_integrity.py`
- `fetch_quarter_scores.py` → `historical_fetch_quarter_scores.py`
- `ingest_elo_ratings.py` → `historical_ingest_elo_ratings.py`
- `ingest_historical_period_odds.py` → `historical_ingest_period_odds.py`

---

# Deletion Candidates (Needs Explicit Approval)
**Proposed cleanup candidates** (non-production artifacts):
- [archive/analysis_snapshots](../archive/analysis_snapshots)
- [archive/predictions](../archive/predictions)
- [archive/slate_outputs](../archive/slate_outputs)
- [data/processed/*](../data/processed)
- [data/raw/github](../data/raw/github) (empty)
- [data/raw/espn](../data/raw/espn) (empty)

These will be removed **only after approval**.

---

# Next Step (Approval Needed)
1) Approve the rename mapping list (or modify it).
2) Confirm deletion candidates.
3) Prediction-only Bicep entry point added in `infra/nba/prediction.bicep` (use this for prediction deployments).
