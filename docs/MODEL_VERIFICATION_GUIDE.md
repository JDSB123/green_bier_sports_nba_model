# Model Verification Guide
**How to Verify Models Are Using Correct Components**

> **Note:** The current production surface only serves the 1H and FG spreads/totals markets. All Moneyline verification steps below are kept for reference only and are not required for the active deployment.

## Problem
Even when the system says it's "ready", you need to verify:
1. Models are actually loaded correctly
2. Predictions use actual models (not simplified calculations)
3. Moneyline uses the audited model (not simplified `0.5 + margin * 0.02`) *(deprecated)*
4. All components are working end-to-end

---

## Quick Verification

### 1. Run Model Integrity Script
```powershell
python scripts/verify_model_integrity.py
```

This checks:
- ✅ All model files exist
- ✅ Models can be loaded
- ✅ Engine initializes correctly
- ✅ Prediction pipeline works
- ✅ Comprehensive edge uses actual models

**Expected Output:**
```
[PASS] ALL CHECKS PASSED - Models are correctly configured and using actual components
```

---

### 2. Check API Verification Endpoint
After restarting the container, call:
```powershell
curl http://localhost:8090/verify
```

This returns:
```json
{
  "status": "pass",
  "checks": {
    "engine_loaded": true,
    "predictors": {
      "spread": true,
      "total": true,
      "moneyline": true
    },
    "moneyline_uses_model": true,
    "moneyline_model_type": "CalibratedClassifierCV",
    "moneyline_has_probabilities": true,
    "moneyline_probabilities_valid": true,
    "test_prediction_works": true,
    "comprehensive_edge_accepts_engine_predictions": true
  },
  "errors": []
}
```

---

### 3. Verify Moneyline Uses Actual Model

**Before Fix:**
- Used simplified calculation: `0.5 + (predicted_margin * 0.02)`
- This was WRONG - not using the audited model

**After Fix:**
- Uses `engine_predictions["full_game"]["moneyline"]["home_win_prob"]`
- This is the ACTUAL model output from the audited spread model

**How to Verify:**
1. Check the comprehensive endpoint code uses `engine_predictions`:
   ```python
   # In src/serving/app.py - comprehensive endpoint
   engine_predictions = app.state.engine.predict_all_markets(...)
   comprehensive_edge = calculate_comprehensive_edge(
       ...,
       engine_predictions=engine_predictions  # ← This passes actual model predictions
   )
   ```

2. Check `calculate_comprehensive_edge` uses them:
   ```python
   # In src/utils/comprehensive_edge.py
   if engine_predictions and engine_predictions.get("full_game", {}).get("moneyline"):
       ml_pred = engine_predictions["full_game"]["moneyline"]
       model_home_prob = float(ml_pred["home_win_prob"])  # ← Uses actual model
   ```

---

## What Was Fixed

### Issue: Simplified Moneyline Calculation
**Location:** `src/utils/comprehensive_edge.py` line 136

**Before:**
```python
# Simplified ML calculation
model_home_prob = 0.5 + (fg_predicted_margin * 0.02)  # WRONG!
```

**After:**
```python
# Use actual engine predictions if available (audited model)
if engine_predictions and engine_predictions.get("full_game", {}).get("moneyline"):
    ml_pred = engine_predictions["full_game"]["moneyline"]
    model_home_prob = float(ml_pred["home_win_prob"])  # CORRECT - uses actual model
```

### Issue: Comprehensive Endpoint Not Passing Predictions
**Location:** `src/serving/app.py` comprehensive endpoint

**Before:**
```python
comprehensive_edge = calculate_comprehensive_edge(
    features=features,
    fh_features=fh_features,
    odds=odds,
    game=game,
    betting_splits=betting_splits,
    edge_thresholds=edge_thresholds
    # Missing: engine_predictions
)
```

**After:**
```python
# Get actual model predictions from engine
engine_predictions = app.state.engine.predict_all_markets(...)

comprehensive_edge = calculate_comprehensive_edge(
    features=features,
    fh_features=fh_features,
    odds=odds,
    game=game,
    betting_splits=betting_splits,
    edge_thresholds=edge_thresholds,
    engine_predictions=engine_predictions  # ← Now passes actual predictions
)
```

---

## Verification Checklist

Before trusting predictions, verify:

- [ ] `python scripts/verify_model_integrity.py` passes all checks
- [ ] `/verify` endpoint returns `"status": "pass"`
- [ ] Moneyline predictions show `"moneyline_model_type": "CalibratedClassifierCV"`
- [ ] Test prediction generates valid probabilities that sum to ~1.0
- [ ] Comprehensive edge function accepts `engine_predictions` parameter
- [ ] No errors in container logs about missing models

---

## Common Errors and Solutions

### Error: "Engine not loaded"
**Solution:** Check container logs:
```powershell
docker logs nba-api
```
Models may be missing from `data/processed/models/`

### Error: "Missing predictors"
**Solution:** Engine initialization failed. Check:
1. All model files exist in `data/processed/models/`
2. Models are valid (not corrupted)
3. Container has access to models directory

### Error: "Moneyline probabilities don't sum to 1.0"
**Solution:** Model output is invalid. This should never happen with correct models.
- Re-run model training
- Verify model files are not corrupted

### Error: "Test prediction failed"
**Solution:** Prediction pipeline is broken.
- Check feature builder is working
- Verify all required features are present
- Check model compatibility

---

## How Models Are Actually Used

### Moneyline Predictor
**Uses:** Spread model's probabilities (as configured)
```python
# In src/prediction/engine.py
self.moneyline_predictor = MoneylinePredictor(
    model=fg_spread_model,  # Uses spread model
    feature_columns=fg_spread_features,
    ...
)
```

**How it works:**
1. Spread model predicts home/away win probabilities
2. Moneyline predictor uses these probabilities
3. Compares to market implied probabilities
4. Calculates edge and recommends bet

**This is CORRECT** - the spread model is calibrated and backtested for moneyline use.

---

## Summary

✅ **Fixed:** Comprehensive analysis now uses actual model predictions  
✅ **Fixed:** Moneyline uses audited model (not simplified calculation)  
✅ **Added:** Verification script to check all components  
✅ **Added:** API endpoint to verify at runtime  

**To verify everything is working:**
1. Run: `python scripts/verify_model_integrity.py`
2. Check: `curl http://localhost:8090/verify`
3. Both should return "pass" status

If either fails, check the errors and fix the underlying issue before using predictions.
