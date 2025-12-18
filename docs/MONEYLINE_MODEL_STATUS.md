# Moneyline Model Status & Path to Accuracy
**Date:** December 18, 2025

## Current Situation

### What We're Using Now:
- **Spread model** converted to moneyline probabilities
- Conversion formula: `P(win) = 1 / (1 + exp(-0.16 * predicted_margin))`
- This is an **approximation**, not a dedicated moneyline model

### Why This Isn't Fully "Correct":
1. **Spread model predicts**: "Will home team cover -5.5?" → 53.6% chance to cover
2. **Moneyline needs**: "Will home team win outright?" → Different question
3. **Conversion is approximate**: The logistic function is a standard approximation, but not as accurate as a model trained specifically for moneyline

---

## When Will It Be "Correct"?

### Answer: When We Train & Use a Dedicated Moneyline Model

The model will be "correct" when:

1. ✅ **We train a dedicated moneyline model** using `scripts/train_models.py`
   - This model predicts win/loss directly (not cover)
   - Uses moneyline-specific features (Elo, Pythagorean, momentum)
   - Is calibrated specifically for moneyline betting

2. ✅ **We save the model** to `models/production/moneyline_model.joblib`

3. ✅ **We update the engine** to load the moneyline model instead of using spread model

4. ✅ **We update MoneylinePredictor** to use the dedicated model instead of conversion

---

## Current Code Status

### What Exists:
- ✅ `MoneylineModel` class (`src/modeling/models.py`) - Can be trained
- ✅ Training script (`scripts/train_models.py`) - Can train moneyline model
- ✅ Moneyline-specific features (Elo, Pythagorean, momentum, etc.)

### What's Missing:
- ❌ **No `moneyline_model.joblib` file** in `models/production/`
- ❌ **Engine still uses spread model** (line 92-97 in `engine.py`)
- ❌ **No loader function** for moneyline model in `src/prediction/models.py`

---

## The Fix I Applied (Temporary)

I fixed the immediate bug (using cover probabilities as win probabilities) by:
- Converting spread cover probabilities to win probabilities using predicted margin
- Using logistic function: `P(win) = 1 / (1 + exp(-k * margin))`

**This is better than before, but still an approximation.**

---

## Path to Full Accuracy

### Step 1: Train Moneyline Model
```bash
python scripts/train_models.py
```
This will:
- Train a `MoneylineModel` on historical win/loss data
- Save to `models/production/moneyline_model.joblib`
- Use moneyline-specific features (not just spread features)

### Step 2: Add Moneyline Model Loader
Add to `src/prediction/models.py`:
```python
def load_moneyline_model(models_dir: Path) -> Tuple[Any, List[str]]:
    """Load full game moneyline model."""
    model_path = models_dir / "moneyline_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Moneyline model not found: {model_path}")
    # Load model and features...
```

### Step 3: Update Engine to Use Moneyline Model
In `src/prediction/engine.py`, change:
```python
# OLD (current):
self.moneyline_predictor = MoneylinePredictor(
    model=fg_spread_model,  # ❌ Using spread model
    ...
)

# NEW (after training):
fg_moneyline_model, fg_moneyline_features = self._load_required_model(
    load_moneyline_model, "Full Game Moneyline"
)
self.moneyline_predictor = MoneylinePredictor(
    model=fg_moneyline_model,  # ✅ Using dedicated moneyline model
    ...
)
```

### Step 4: Update MoneylinePredictor
In `src/prediction/moneyline/predictor.py`, change:
```python
# OLD (current conversion):
spread_proba = self.model.predict_proba(X)[0]
home_cover_prob = float(spread_proba[1])
# Convert to win prob using margin...

# NEW (after training):
ml_proba = self.model.predict_proba(X)[0]
home_win_prob = float(ml_proba[1])  # Direct win probability!
away_win_prob = float(ml_proba[0])
```

---

## Is It Just a Function of Spread Calculation?

**No, it's more than that.**

### Current (Approximation):
- Uses spread model → converts to win probability
- Works reasonably well, but not as accurate as dedicated model

### Future (Dedicated Model):
- Uses moneyline model → direct win probability
- Trained specifically on win/loss outcomes
- Uses moneyline-specific features (Elo, Pythagorean, momentum)
- More accurate for moneyline betting

### Why Dedicated Model is Better:
1. **Different target**: Win/loss vs cover
2. **Different features**: Moneyline-specific features (Elo, Pythagorean)
3. **Different calibration**: Calibrated for moneyline probabilities
4. **Better accuracy**: Historical backtests show ~65.5% accuracy for dedicated moneyline model vs ~60.6% for spread model

---

## Summary

**Current Status:**
- ✅ Fixed bug (no longer using cover probabilities directly)
- ✅ Using conversion formula (approximation)
- ⚠️ Not using dedicated moneyline model yet

**When Will It Be "Correct"?**
- When we train and deploy a dedicated moneyline model
- The infrastructure exists, we just need to:
  1. Train the model
  2. Add loader function
  3. Update engine to use it
  4. Update predictor to use direct probabilities

**Is It Just Spread Calculation?**
- No, it's more than that
- Currently: Spread model → conversion → moneyline
- Future: Dedicated moneyline model → direct moneyline predictions
- The dedicated model will be more accurate because it's trained specifically for win/loss outcomes
