# Model Architecture

**Last Updated:** 2026-01-23
**Status:** Consolidated from MODEL_VERIFICATION_GUIDE, FEATURE_ARCHITECTURE_v33.1.0

---

## Markets (4 Independent Models)

| Period | Spread | Total |
|--------|--------|-------|
| **1H** (First Half) | 1h_spread | 1h_total |
| **FG** (Full Game) | fg_spread | fg_total |

Each model is trained and evaluated independently.

---

## Unified Feature Schema

All models use **unified feature names** for consistency.

The difference between 1H and FG is the **values**, not the names:

```python
# Same feature name, different values
FG Model: "home_ppg" = 115.0  # Full game average
1H Model: "home_ppg" = 57.5   # First half average
```

---

## Feature Categories

### Period-Specific Features (Different values for 1H vs FG)

| Category | Features |
|----------|----------|
| **Scoring** | `home_ppg`, `away_ppg`, `home_papg`, `away_papg`, `ppg_diff` |
| **Margins** | `home_margin`, `away_margin` |
| **Win Rates** | `home_win_pct`, `away_win_pct` |
| **Pace** | `home_pace`, `away_pace`, `expected_pace` |
| **Form** | `home_l5_margin`, `away_l5_margin`, `home_l10_margin`, `away_l10_margin` |
| **Efficiency** | `home_ortg`, `home_drtg`, `home_net_rtg`, `away_ortg`, `away_drtg`, `away_net_rtg` |

### Shared Features (Same for both periods)

| Category | Features |
|----------|----------|
| **Rest** | `home_rest`, `away_rest`, `rest_diff`, `home_b2b`, `away_b2b` |
| **Travel** | `away_travel_distance`, `away_timezone_change`, `away_travel_fatigue` |
| **Injuries** | `home_injury_impact_ppg`, `away_injury_impact_ppg`, `home_star_out`, `away_star_out` |
| **Elo** | `home_elo`, `away_elo`, `elo_diff`, `elo_prob_home` |
| **Betting** | `public_home_pct`, `sharp_money_side`, `rlm_indicator`, `consensus_spread` |

---

## Feature Mapping (1H Models)

At prediction time, 1H models need 1H-specific data mapped to unified names:

```python
# Feature engineering creates both versions
features = {
    "home_ppg": 115.0,      # FG average
    "home_ppg_1h": 57.5,    # 1H average
}

# For 1H prediction, map 1H values to FG names
if predicting_1h:
    mapped = map_1h_features_to_fg_names(features)
    # Result: {"home_ppg": 57.5}  # Now has 1H value
```

Function: `map_1h_features_to_fg_names()` in `src/prediction/engine.py`

---

## Model Files

Location: `models/production/`

| File | Purpose |
|------|---------|
| `fg_spread_model.joblib` | FG spread predictions |
| `fg_total_model.joblib` | FG total predictions |
| `1h_spread_model.joblib` | 1H spread predictions |
| `1h_total_model.joblib` | 1H total predictions |
| `model_pack.json` | Version metadata |
| `feature_importance.json` | Feature rankings |

---

## Verification

### Script Verification

```bash
python scripts/model_validate.py
```

Checks:
- ✅ All model files exist
- ✅ Models can be loaded
- ✅ Engine initializes correctly
- ✅ Prediction pipeline works

Expected: `[PASS] ALL CHECKS PASSED`

### API Verification

```bash
curl http://localhost:8090/verify
```

Returns:
```json
{
  "status": "pass",
  "checks": {
    "engine_loaded": true,
    "predictors": {"spread": true, "total": true},
    "test_prediction_works": true
  }
}
```

---

## Feature Building

### Runtime Builder

`RichFeatureBuilder.build_game_features()` in `src/features/rich_features.py`

Fetches live data from APIs and computes features for prediction.

### Training Schema

`UNIFIED_FEATURE_NAMES` in `src/modeling/unified_features.py`

Defines the 102-feature training schema.

### Actual Model Inputs

From trained models (`.joblib` files):
- FG spread: 35 features
- FG total: 35 features
- 1H spread: 40 features
- 1H total: 40 features

---

## Prediction Flow

```
Request → RichFeatureBuilder → Feature Dict → Model.predict() → Predictions
                                    ↓
                           (1H: map to FG names)
```

---

## Training

```bash
python scripts/model_train_all.py
```

Uses `data/processed/training_data.csv` (canonical training data).

---

## Common Issues

### "Engine not loaded"
Check container logs. Models may be missing from `models/production/`.

### "Missing predictors"
Model initialization failed. Verify model files exist and are valid.

### "Test prediction failed"
Feature builder or model incompatibility. Check feature names match.
