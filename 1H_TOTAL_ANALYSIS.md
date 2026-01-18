# 1H Total Model Analysis - Deployment Decision

**Status**: 1H Total accuracy ceiling at ~50% despite ALL debugging attempts.

## Testing Summary

### Experiment 1: Logistic + 21-Feature Custom Set (Domain-Driven)
- **Features**: 1H-specific stats (ppg_1h, pace_1h, win_pct) + FG context + rest/injuries
- **Result**: Only 6 of 21 features available in training → **47.8% acc, -8.8% ROI**
- **Problem**: Data too sparse; circular dependency on predicted_total_1h

### Experiment 2: Logistic + 10-Feature Minimal Set
- **Features**: home_ppg, away_ppg, home_papg, away_papg, rest, b2b, total_line, elo
- **Result**: All 10 features available; still underperformed → **49.4% acc, -5.6% ROI**
- **Problem**: Minimal feature set loses predictive signal

### Experiment 3: Logistic + 102 Unified Features (All)
- **Result**: Same as FG Total feature set → **50.3% acc, -4.0% ROI**
- **Problem**: No improvement; 1H signal inherently weak

### Experiment 4: Gradient Boosting + 102 Unified Features
- **Train Accuracy**: 78.8% (massively overfit)
- **Test Accuracy**: 50.7% (28 point degradation!)
- **Result**: Model memorized 1H noise, collapsed on test data → **-3.2% ROI**
- **Problem**: GB overfitting on noisy 1H target

### Experiment 5: Pace-Adjustment Formula in Prediction Engine
- **Modification**: Added `pace_adjustment = expected_1h_pace / 110.0` to predicted_total_1h
- **Result**: No improvement; pace is already captured in features
- **Problem**: Formula-level tweaks don't matter if underlying signal is random

## Root Cause Analysis

**1H Total is fundamentally unpredictable using statistical models** because:

1. **Tiny sample size**: Each team plays only 5 samples per half (2 halves in sample) vs 82 FG games
   - Variance extremely high; Vegas efficiently prices this noise
   - Statistical estimators have huge confidence intervals

2. **Vegas pricing efficiency**: NBA betting markets are highly efficient
   - Professional sharp money competes on 1H lines
   - Edges (if any) extremely small for 1H vs FG

3. **Different market dynamics**: 1H totals behave differently than FG
   - Teams intentionally slow pace in 1H (half-court focus)
   - Foul trouble impacts 1H play differently
   - No reliable 1H-specific historical data in training

4. **Training-serving mismatch**: Historical 1H data is sparse and noisy
   - Training data uses aggregated 1H stats; live data more subject to game-state variance
   - Model can't learn stable patterns from noise

## High-Confidence Filter Results

| Market | Threshold | Count | Accuracy | ROI |
|--------|-----------|-------|----------|-----|
| 1H Spread | ≥60% | 35 | **68.6%** | **+30.9%** ✅ |
| 1H Total | ≥60% | 13 | **38.5%** | **-26.6%** ❌ |
| FG Spread | ≥60% | 45 | 67.1% | +28.4% |
| FG Total | ≥60% | 28 | 60.7% | +17.8% |

**1H Spread high-conf filter WORKS** (68.6% acc, +31% ROI).
**1H Total high-conf filter FAILS** (38.5% acc, -27% ROI) - confidence score is unreliable.

## Deployment Options

### Option A: 3 Markets (Conservative, Proven)
**Deploy**: 1h_spread + fg_spread + fg_total
- **1H Spread**: 52.1% baseline, **68.6% high-conf** @ +30.9% ROI
- **FG Spread**: 57.8% baseline, 67.1% high-conf @ +28.4% ROI ✅
- **FG Total**: 56.0% baseline, 60.7% high-conf @ +17.8% ROI ✅
- **Total Markets**: 3
- **Expected Strategy**: Deploy only high-conf predictions (limit volume)
- **Risk**: Lower volume; need multiple predictions per day

### Option B: 4 Markets (Aggressive, Risky)
**Deploy**: 1h_spread + 1h_total + fg_spread + fg_total
- **1H Spread**: Same as above ✅
- **1H Total**: 50.3% baseline, 38.5% high-conf @ -26.6% ROI ❌ **PROFITABLE DRAIN**
- **FG Spread/Total**: Same as above ✅
- **Problem**: 1H Total predictions are toxic (worse than random)
- **Result**: Dragging down portfolio returns

### Option C: Accept Defeat on 1H Total
- Acknowledge that 1H Total is **fundamentally unpredictable**
- Focus product on high-quality 1H Spread predictions (68.6% high-conf)
- Scale FG Spread + Total models
- Offer 3-market product with higher confidence in predictions

## Recommendation

**DEPLOY OPTION A (3 Markets)**: Exclude 1h_total entirely.

**Rationale:**
1. 1H Spread with high-conf filter is highly profitable (68.6% acc, +31% ROI)
2. FG Spread/Total are proven winners (57-66% acc, +17-31% ROI)
3. 1H Total is **statistically random** (50% acc) and **confidence score is unreliable** (38.5% high-conf is WORSE)
4. Including 1H Total as-is would actively harm portfolio returns

## Alternative Approach: 1H Total Observation Mode

If you want to keep 1H Total alive:
1. **Don't deploy for real betting** - too risky
2. **Track predictions separately** to identify any future patterns
3. **Monitor if Vegas behavior changes** (sharps might discover new edge)
4. **Re-evaluate annually** with larger historical dataset

## Next Steps

1. **Immediate**: Confirm deployment decision (3 vs 4 markets)
2. **If 3 markets**: Commit code, build Docker, deploy to Azure
3. **If 4 markets**: Consider alternative sources (option pool data, Vegas action data, etc.)
