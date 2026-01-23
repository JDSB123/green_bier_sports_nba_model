# Model Verification Guide
**How to Verify Models Are Using Correct Components**

> **Note:** The current production surface only serves the 1H and FG spreads/totals markets.

## Problem
Even when the system says it's "ready", you need to verify:
1. Models are actually loaded correctly
2. Predictions use actual models (not simplified calculations)
3. All components are working end-to-end

---

## Quick Verification

### 1. Run Model Integrity Script
```powershell
python scripts/model_validate.py
```

This checks:
- All model files exist
- Models can be loaded
- Engine initializes correctly
- Prediction pipeline works
- Comprehensive edge uses actual models

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
      "total": true
    },
    "test_prediction_works": true,
    "comprehensive_edge_accepts_engine_predictions": true
  },
  "errors": []
}
```

---

## Verification Checklist

Before trusting predictions, verify:

- [ ] `python scripts/model_validate.py` passes all checks
- [ ] `/verify` endpoint returns `"status": "pass"`
- [ ] Test prediction generates valid outputs
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

### Error: "Test prediction failed"
**Solution:** Prediction pipeline is broken.
- Check feature builder is working
- Verify all required features are present
- Check model compatibility

---

## Summary

- Verification focuses on spread/total markets only
- Integrity script and `/verify` endpoint should both pass
- Fix any errors before using predictions
