# Betting Splits / RLM Feature Status

**Last Updated:** Jan 5, 2026
**Status:** ✅ Included in models for live predictions

## Current State

The prediction models **include RLM and betting splits features** for live predictions:

1. **Models trained with features** - 55 features including RLM/splits
2. **Live predictions fetch real data** - Action Network provides real splits
3. **Training data limitation** - Historical data has placeholders (50/50)

### How It Works

| Phase | Data Source | RLM/Splits Values |
|-------|-------------|-------------------|
| **Training** | Historical CSV | Placeholder (50/50, constant) |
| **Live Prediction** | Action Network API | Real betting splits |

The model learns coefficients for these features even with constant training data.
When live data provides **real variation**, the model can use these signals.

### Feature Importance Note

During training, these features show 0% importance because training data is constant:

| Feature | Training Importance | Live Potential |
|---------|---------------------|----------------|
| `is_rlm_spread` | 0.00% | ✅ Will vary |
| `sharp_side_spread` | 0.00% | ✅ Will vary |
| `spread_public_home_pct` | 0.00% | ✅ Will vary |
| `spread_ticket_money_diff` | 0.00% | ✅ Will vary |

## Live Prediction Flow

```
1. Fetch game info (teams, time)
2. Fetch odds from The Odds API
3. Fetch betting splits from Action Network  <-- REAL DATA
4. Compute features (including RLM detection)
5. Model prediction with full 55 features
```

The `src/ingestion/betting_splits.py` module handles:
- `fetch_splits_action_network()` - Action Network public API
- `fetch_public_betting_splits()` - Auto-selects best source
- `detect_reverse_line_movement()` - Calculates RLM signals
- `splits_to_features()` - Converts to model feature format

## Improving Model Accuracy with Real Training Data

To fully leverage RLM/splits, collect historical data for future retraining:

### Daily Collection (Recommended)

```bash
# Run daily before games start
python scripts/collect_betting_splits.py --save --append-history
```

This appends to `data/splits/historical_splits.csv` for future model training.

### After 2+ Months of Collection

```bash
# Rebuild training data with real splits
python scripts/build_fresh_training_data.py --include-splits

# Retrain models
python scripts/train_models.py --market all
```

## Files

| File | Purpose |
|------|---------|
| `src/ingestion/betting_splits.py` | Splits fetching infrastructure |
| `src/modeling/unified_features.py` | Feature definitions (includes RLM) |
| `scripts/collect_betting_splits.py` | Daily collection script |
| `data/splits/historical_splits.csv` | Cumulative training data |

## Summary

- ✅ Models include RLM/splits features
- ✅ Live predictions fetch real data from Action Network
- ⚠️ Training used placeholders (retraining will improve accuracy)
- 📊 Collect daily splits for future model improvement
