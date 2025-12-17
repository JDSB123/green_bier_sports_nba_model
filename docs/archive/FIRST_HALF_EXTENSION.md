# First Half Markets - Module Code Only

## Status: ❌ NOT PRODUCTION READY

**Date:** December 17, 2025
**Module Code:** Complete
**Production Status:** NOT integrated, NOT tested, NOT validated

**DO NOT USE IN PRODUCTION** - This is developer documentation for future integration.

---

## What Was Added

### 1. First Half Filters ✅
```python
# src/prediction/filters.py

FirstHalfSpreadFilter(
    filter_small_spreads=True,
    small_spread_min=1.5,  # ~50% of FG 3.0
    small_spread_max=3.0,  # ~50% of FG 6.0
    min_edge_pct=0.05,
)

FirstHalfTotalFilter(
    use_filter=False,  # No filtering (baseline best)
    min_edge_pct=0.05,
)
```

### 2. First Half Prediction Methods ✅
```python
# src/prediction/predictor.py

class PredictionEngine:
    # New methods:
    def predict_first_half_spread(features, first_half_spread_line)
    def predict_first_half_total(features, first_half_total_line)
    def predict_game_all_markets(features, spread_line, total_line, fh_spread_line, fh_total_line)
```

### 3. Updated Exports ✅
```python
# src/prediction/__init__.py
from src.prediction import (
    FirstHalfSpreadFilter,  # NEW
    FirstHalfTotalFilter,   # NEW
    PredictionEngine,       # Updated with 1H methods
)
```

---

## How It Works

### Using FG Models for 1H Predictions

**Current Approach:**
- Uses existing Full Game models
- Scales predictions to ~50% of FG values
- Applies proportional filtering thresholds

**Example:**
```
Full Game Predicted Margin: +6.0 pts (home favored)
  → First Half Predicted Margin: +3.0 pts (50% scaled)

Full Game Predicted Total: 234.0 pts
  → First Half Predicted Total: 117.0 pts (50% scaled)
```

### Filtering Logic

**1H Spreads:**
- Filter out 1.5-3.0 point spreads (proportional to FG 3-6 pt filter)
- Require 5% model edge (same as FG)

**1H Totals:**
- No filtering (like FG totals, baseline is best)
- Optionally can enable 5% edge filter

---

## Usage Examples

### Basic 1H Prediction
```python
from src.prediction import PredictionEngine

engine = PredictionEngine(models_dir)

# Predict first half spread
fh_spread_pred = engine.predict_first_half_spread(
    features=game_features,
    first_half_spread_line=-3.5,  # 1H line
)

print(f"1H Spread: {fh_spread_pred['bet_side']} with {fh_spread_pred['confidence']:.1%} confidence")
print(f"Passes filter: {fh_spread_pred['passes_filter']}")

# Predict first half total
fh_total_pred = engine.predict_first_half_total(
    features=game_features,
    first_half_total_line=117.5,  # 1H line
)

print(f"1H Total: {fh_total_pred['bet_side']} {fh_total_pred['predicted_total']:.1f}")
```

### All Markets at Once
```python
# Predict all 4 markets (FG + 1H)
all_preds = engine.predict_game_all_markets(
    features=game_features,
    spread_line=-7.5,           # FG spread
    total_line=235.0,           # FG total
    first_half_spread_line=-3.5, # 1H spread
    first_half_total_line=117.5,  # 1H total
)

# Access predictions
fg_spread = all_preds["full_game"]["spread"]
fg_total = all_preds["full_game"]["total"]
fh_spread = all_preds["first_half"]["spread"]
fh_total = all_preds["first_half"]["total"]
```

### Custom Filters
```python
from src.prediction import (
    PredictionEngine,
    FirstHalfSpreadFilter,
    FirstHalfTotalFilter,
)

# More conservative 1H spread filter
fh_spread_filter = FirstHalfSpreadFilter(
    filter_small_spreads=True,
    small_spread_min=1.0,  # Wider filter range
    small_spread_max=4.0,
    min_edge_pct=0.07,     # Higher edge requirement
)

# Enable filtering on 1H totals (not recommended)
fh_total_filter = FirstHalfTotalFilter(
    use_filter=True,
    min_edge_pct=0.06,
)

engine = PredictionEngine(
    models_dir,
    first_half_spread_filter=fh_spread_filter,
    first_half_total_filter=fh_total_filter,
)
```

---

## Performance Expectations

### Estimated Performance (Not Backtested Yet)

Since we're using FG models with 50% scaling:

**1H Spreads:**
- Expected: Similar to FG spreads (60.6% accuracy with filtering)
- Scaling may introduce some error (+/- 2-3%)
- Conservative estimate: 57-63% accuracy

**1H Totals:**
- Expected: Similar to FG totals (59.2% accuracy baseline)
- Scaling may be more reliable for totals
- Conservative estimate: 56-62% accuracy

### Future Improvements

**When to train dedicated 1H models:**
- After accumulating 500+ games with 1H lines
- Expected improvement: +2-5% accuracy vs scaled FG models
- Would allow for 1H-specific features (1Q scoring, pace, etc.)

**Priority for dedicated models:** Medium-Low
- Current scaling approach provides reasonable baseline
- Focus first on FG model improvements
- Train 1H models once FG performance stable

---

## Integration with Production Script

### Option 1: Add --include-first-half flag to predict_v2.py
```bash
python scripts/predict_v2.py --include-first-half
```

### Option 2: Create predict_v3.py with all markets
```bash
python scripts/predict_v3.py  # FG + 1H by default
```

### Option 3: Separate script for 1H only
```bash
python scripts/predict_first_half.py
```

**Recommendation:** Add flag to predict_v2.py for flexibility

---

## Technical Details

### Scaling Approach
```python
# First Half Margin
predicted_margin_1h = fg_predicted_margin * 0.5

# First Half Total
predicted_total_1h = fg_predicted_total * 0.5
```

**Why 50%?**
- Historical NBA data shows 1H scores ~48-52% of FG totals
- 50% is a reasonable default approximation
- Could be refined with team-specific pace analysis

### Model Reuse
```python
# Same FG spread model used for both
fg_spread = spread_model.predict_proba(X)[0]   # Full game
fh_spread = spread_model.predict_proba(X)[0]   # First half (same model!)

# Only the interpretation changes (via scaling)
```

**Benefits:**
- No need to train new models
- Consistent model behavior
- Easy to switch to dedicated models later

**Limitations:**
- Doesn't capture 1H-specific dynamics
- Scaling factor is fixed at 50%
- No 1H-specific features (1Q performance, etc.)

---

## Example Output (Conceptual)

```
Game: Memphis Grizzlies @ Minnesota Timberwolves
Game Time: Wed Dec 17, 07:10 PM CST

FULL GAME:
  Spread: MIN -7.5 → Bet HOME (56.4% conf, +10.8 edge) ✅
  Total: 232.5 → Bet OVER (91.2% conf, +2.0 edge) ✅

FIRST HALF:
  Spread: MIN -3.5 → Bet HOME (56.4% conf, +5.4 edge) ✅
  Total: 116.5 → Bet OVER (91.2% conf, +1.0 edge) ✅

RECOMMENDED PLAYS: 4 (2 FG, 2 1H)
```

---

## Backtest TODO

### To validate 1H performance:
1. Collect historical 1H lines from The Odds API
2. Create backtest_first_half.py script
3. Test on 200+ games
4. Compare vs FG performance
5. Refine filtering thresholds based on results

### Expected Timeline:
- Data collection: 1-2 weeks
- Backtest implementation: 1 day
- Analysis: 1 day
- Filter tuning: 1 day

---

## Files Modified

```
src/prediction/
├── __init__.py         ✅ Added 1H filter exports
├── filters.py          ✅ Added FirstHalfSpreadFilter, FirstHalfTotalFilter
└── predictor.py        ✅ Added predict_first_half_spread(), predict_first_half_total(), predict_game_all_markets()

FIRST_HALF_EXTENSION.md ✅ This file
```

---

## Quick Reference

### Filtering Thresholds

| Market | Filter Range | Min Edge | Recommended |
|--------|--------------|----------|-------------|
| **FG Spread** | 3.0-6.0 pts | 5% | ✅ Use filter |
| **FG Total** | None | N/A | ✅ No filter |
| **1H Spread** | 1.5-3.0 pts | 5% | ✅ Use filter |
| **1H Total** | None | N/A | ✅ No filter |

### Scaling Factors

| Metric | FG Value | 1H Value (Scaled) |
|--------|----------|-------------------|
| Predicted Margin | +6.0 | +3.0 (50%) |
| Predicted Total | 234.0 | 117.0 (50%) |
| Spread Line | -7.5 | -3.75 (typical) |
| Total Line | 235.0 | 117.5 (typical) |

---

## Summary

**What We Built:**
- ✅ First half spread + total prediction support
- ✅ Proportional filtering (1.5-3pt spreads, no total filter)
- ✅ Clean modular extension
- ✅ Uses existing FG models with 50% scaling

**What's Next:**
- 🔲 Add 1H support to predict_v2.py (or create predict_v3.py)
- 🔲 Collect 1H historical lines
- 🔲 Backtest 1H performance
- 🔲 Consider training dedicated 1H models (future)

**Production Ready:** ❌ NO (see "What Needs to Happen Before Production" below)

---

## What Needs to Happen Before Production

### Before 1H is Ready:
1. ❌ **Script Integration** - Add 1H support to predict_v2.py
   - Extract 1H lines from The Odds API
   - Call predict_game_all_markets()
   - Add 1H bets to betting card

2. ❌ **Live Testing** - Test on real games
   - Verify 1H line extraction works
   - Confirm predictions make sense
   - Check filtering behaves correctly

3. ❌ **Backtest Validation** - Validate on historical data
   - Collect 200+ games with 1H lines
   - Run backtest_first_half.py (needs to be created)
   - Validate ~57-63% accuracy estimate
   - Tune filtering thresholds if needed

4. ❌ **Documentation Update** - Update production docs
   - Add 1H to PRODUCTION_READY.md
   - Update README.md with 1H usage
   - Document expected performance

**Estimated Effort:** 2-3 days of work

**Current Status:** ❌ Module code only, NOT production ready

---

**Built:** December 17, 2025
**Status:** Developer reference only - DO NOT USE IN PRODUCTION
**Next Steps:** Complete items 1-4 above before using
