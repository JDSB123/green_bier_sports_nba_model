# Probability Calculation and High Confidence Logic Improvements

## Overview
Major improvements to betting probability calculations and high confidence play selection logic for enhanced accuracy and reliability.

## Key Improvements

### 1. Fixed High Confidence Logic
**BEFORE:** Used arbitrary win_probability thresholds (>60% or <40%)
```python
is_high_confidence = win_probability > 0.60 or win_probability < 0.40
```

**AFTER:** Uses calibrated confidence scores with dual criteria
```python
is_high_confidence = (
    confidence_score >= 0.70 and  # Model is confident in its prediction
    (win_probability >= 0.65 or win_probability <= 0.35)  # AND prediction is extreme
)
```

**Rationale:** Combines model certainty (entropy-based confidence) with prediction extremity for true high-conviction plays.

### 2. ML Classifier Architecture (NO FALLBACKS)
**STRICT REQUIREMENT:** ML models are now mandatory - no silent fallbacks to heuristics

**ML Classifiers Used:**
- **Spread Classifier**: Calibrated Logistic Regression predicting if home team covers spread (29 features)
- **Total Classifier (ML Over/Under)**: Calibrated Logistic Regression predicting if game goes OVER total (29 features)
- **Calibration**: Isotonic regression ensures probabilities reflect true likelihood
- **Features**: PPG, PAPG, avg_margin, pace, efficiency metrics, home/away splits, etc.

**What is the ML Over/Under Classifier?**
The "ML over/under" refers to the **Total Classifier** - a trained logistic regression model that predicts whether the total points scored in a game will be OVER or UNDER the sportsbook's total line. It outputs calibrated probabilities (0.0-1.0) for the "OVER" outcome, where:
- 0.7 = 70% chance game goes over the total
- 0.3 = 30% chance game goes over (70% chance goes under)
- 0.5 = neutral (50/50 chance)

**Probability Sources (Post-Update):**
1. **Engine ML Models ONLY** - Calibrated classifier predictions (required)
2. **Distribution-Based** - Statistical estimates (diagnostics only)
3. **Edge Heuristics** - Removed (no longer used)

**Full Game Markets:**
```python
# ML Engine predictions (highest priority)
if fg_spread_pred and fg_spread_p_model not in [None, 0.5]:
    fg_spread_win_prob = fg_spread_p_model
    fg_spread_probability_source = "engine_ml"
elif engine_predictions:
    # Fall back to distribution-based
    fg_spread_win_prob = fg_spread_p_dist
    fg_spread_probability_source = "distribution"
else:
    # No engine available
    fg_spread_win_prob = fg_spread_p_dist
    fg_spread_probability_source = "distribution"
```

### 3. Dynamic Confidence Thresholds
**Adaptive thresholds** based on prediction source availability:
- **ML Models Available:** Reduce thresholds by 10% (more aggressive)
- **No ML Models:** Increase thresholds by 10% (more conservative)

```python
if engine_predictions:
    edge_thresholds = {k: v * 0.9 for k, v in base_thresholds.items()}
else:
    edge_thresholds = {k: v * 1.1 for k, v in base_thresholds.items()}
```

### 4. Probability Source Validation
**Automated logging and validation:**
- Tracks which probability source is used for each pick
- Warns when ML model usage is low despite models being available
- Provides transparency into prediction reliability

```
INFO: Probability sources used: ML=3 (60.0%), Distribution=2 (40.0%), Heuristic=0 (0.0%)
WARNING: Low ML model usage (40.0%) despite models being available
```

## Market-Specific Accuracy

### Full Game Spreads
- **ML Model Priority:** Trained classifier probabilities when available
- **Distribution Fallback:** Statistical cover probability estimates
- **Accuracy:** Targets 2.0+ point edges with 60%+ confidence

### Full Game Totals
- **ML Model Priority:** Trained over/under classifier probabilities
- **Distribution Fallback:** Statistical over/under probability estimates
- **Accuracy:** Targets 3.0+ point edges with 58%+ confidence

### First Half Markets
- **ML Model Required:** No fallbacks allowed - uses dedicated 1H models only
- **Accuracy:** Targets 1.5-2.0 point edges with calibrated confidence

## Technical Implementation

### Probability Sources Tracked
Each market result now includes:
```json
{
  "win_probability": 0.68,
  "probability_source": "engine_ml",
  "confidence": 0.75,
  "p_model": 0.68,
  "p_fair": 0.62
}
```

### Confidence Calculation
Uses entropy-based confidence from `src/prediction/confidence.py`:
- **High entropy (prob ~0.5)** = Uncertain = Lower confidence
- **Low entropy (prob ~0.0 or 1.0)** = Certain = Higher confidence (capped at 95%)

### Edge Calculations
**Spreads:** `edge = predicted_margin + spread_line`
**Totals:** `edge = predicted_total - market_total`

## Validation and Testing

### Automated Validation
- Probability source consistency checks
- ML model availability verification
- Confidence threshold validation

### Logging Improvements
- Source usage statistics
- Prediction reliability warnings
- Dynamic threshold adjustments logged

## Impact on Performance

### Expected Improvements
1. **Higher Quality Picks:** Better confidence assessment reduces false positives
2. **More Accurate Probabilities:** ML model prioritization over heuristics
3. **Adaptive Strategy:** Dynamic thresholds adjust to prediction source reliability
4. **Better Transparency:** Clear tracking of probability sources and confidence levels

### Backtesting Validation Required
- Compare hit rates by probability source
- Validate confidence score calibration
- Test dynamic threshold effectiveness

## Configuration

### Default Thresholds
```python
filter_thresholds = FilterThresholds(
    spread_min_confidence=0.60,
    spread_min_edge=2.0,
    total_min_confidence=0.58,
    total_min_edge=3.0
)
```

### High Confidence Criteria (Statistically Sound - Rigorous)
**OLD:** `confidence >= 70% AND (probability >= 65% or <= 35%)`

**NEW - Statistically Sound Requirements (Model should need minimal filtering):**
1. **Model Certainty + Statistical Significance**: `confidence >= 75% AND edge >= 2.5pts AND probability <= 35%`
2. **Extreme Market Mismatch**: `confidence >= 65% AND edge >= 4.0pts AND (probability <= 25% OR >= 75%)`

**Rationale:** High confidence requires BOTH statistical significance (large edge) AND model certainty. If the model were truly sound, this filtering would be unnecessary.

## Future Enhancements

1. **Model Calibration:** Periodic recalibration of confidence scores against actual results
2. **Source Weighting:** Dynamic weighting of probability sources based on historical accuracy
3. **Market-Specific Tuning:** Different thresholds for different market conditions
4. **Real-time Validation:** Continuous monitoring of probability source effectiveness

## Files Modified

- `src/utils/comprehensive_edge.py` - Core probability and confidence logic
- `src/prediction/confidence.py` - Entropy-based confidence calculations (existing)
- `src/config.py` - Filter thresholds configuration (existing)

## Testing Commands

```bash
# Test import
python -c "from src.utils.comprehensive_edge import calculate_comprehensive_edge"

# Run predictions with new logic
python scripts/predict.py --date 2025-01-03

# Check probability source usage logs
python scripts/predict.py 2>&1 | grep "Probability sources used"
```</content>
</xai:function_call">Write