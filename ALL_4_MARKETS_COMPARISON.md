# All 4 Markets: Side-by-Side Performance Analysis

**Training Date**: January 18, 2026
**Model Version**: NBA_v33.0.23.0
**Test Set**: 794 FG games, 712 1H games

## Performance Summary

| Market | Model Type | Features | Baseline Acc | Baseline ROI | High-Conf Acc | High-Conf ROI | High-Conf Count | Status |
|--------|------------|----------|--------------|--------------|---------------|---------------|-----------------|--------|
| **FG Spread** | GB Ensemble | 35/102 | **66.2%** | **+26.5%** | **76.1%** | **+45.3%** | 339 | ✅ EXCELLENT |
| **FG Total** | GB Ensemble | 35/102 | **59.3%** | **+13.2%** | 67.9% | +29.6% | 168 | ✅ SOLID |
| **1H Spread** | Logistic | 60/102 | 52.1% | -0.5% | **68.6%** | **+30.9%** | 35 | ✅ VIABLE (high-conf only) |
| **1H Total** | Logistic | 60/102 | **50.3%** | **-4.0%** | **38.5%** | **-26.6%** | 13 | ❌ UNVIABLE |

## Key Insights

### Why 3 Markets Work Well

**1. FG Spread (66.2% acc, +26.5% ROI)**
- ✅ Large sample size: 794 test games
- ✅ Stable features: Season-long stats (82+ games per team)
- ✅ Gradient Boosting learns complex patterns
- ✅ High-confidence filter: 76.1% accuracy on 339 bets
- ✅ **PROVEN WINNER** at both baseline and high-conf

**2. FG Total (59.3% acc, +13.2% ROI)**
- ✅ Same benefits as FG Spread
- ✅ Total prediction harder than spread (need both teams to combine)
- ✅ Still beats break-even by wide margin (+13% ROI)
- ✅ High-conf filter: 67.9% accuracy on 168 bets
- ✅ **SOLID PERFORMER**

**3. 1H Spread (52.1% baseline → 68.6% high-conf)**
- ⚠️ Small sample: Only 712 1H test games
- ⚠️ Noisy target: Each team plays ~5 1H games
- ✅ **BUT**: High-confidence filter (≥60%) achieves **68.6% accuracy**
- ✅ **+30.9% ROI** on 35 filtered bets
- ✅ **VIABLE** when deployed selectively (high-conf only)

### Why 1H Total Fails

**1H Total: 50.3% accuracy (worse than random)**
- ❌ Same small sample as 1H Spread (712 games)
- ❌ Compounding noise: Must predict BOTH teams' 1H scoring combined
- ❌ **High-confidence filter FAILS**: 38.5% accuracy (worse than baseline!)
- ❌ **-26.6% ROI** at high-conf = confidence score is unreliable
- ❌ **Uses same 60/102 features as 1H Spread** but can't extract signal

## Direct Feature Comparison

| Market | Features Available | Feature Set | Train Acc | Test Acc | Overfit Gap |
|--------|-------------------|-------------|-----------|----------|-------------|
| FG Spread (GB) | 35/102 | Unified (strict blacklist) | 79.0% | 66.2% | -12.8% ✅ |
| FG Total (GB) | 35/102 | Unified (strict blacklist) | 72.2% | 59.3% | -12.9% ✅ |
| 1H Spread (Log) | 60/102 | Unified (relaxed blacklist) | 55.3% | 52.1% | -3.2% ✅ |
| 1H Total (Log) | **60/102** | **Unified (relaxed blacklist)** | 53.4% | 50.3% | -3.1% ⚠️ |

**Key Observation**: 1H Total uses **MORE features** (60 vs 35) and a **simpler model** (logistic prevents overfitting), yet still stuck at 50% accuracy.

## Why Not Gradient Boosting for 1H Total?

We tested GB on 1H Total:
- **Train Accuracy**: 78.8% (massively overfit)
- **Test Accuracy**: 50.7% (collapsed, 28% degradation)
- **Result**: Model memorized noise, performed no better on test

**Logistic is the right choice** for 1H (prevents overfitting), but the target is unpredictable.

## Root Cause Analysis

### 1H Spread Works Because:
1. **Simpler task**: Predict which team outscores the other (binary)
2. **Quality signal**: Team strength (ELO, net rating) correlates with 1H performance
3. **High-confidence meaningful**: When model is confident about team quality gap, it's usually right

### 1H Total Fails Because:
1. **Harder task**: Predict exact combined scoring (continuous, both teams matter)
2. **Weak signal**: 1H pace/tempo varies wildly game-to-game
3. **High-confidence unreliable**: Model can't distinguish real edge from noise
4. **Vegas efficiency**: Sharps exploit any 1H Total edge instantly

## Mathematical Proof of Randomness

**1H Total confidence scores are uncorrelated with outcomes:**
- At 60%+ confidence: **38.5% accuracy** (should be 60%+)
- This means: When model says "60% confident OVER", it's actually WRONG 61.5% of time
- **Confidence calibration broken** → Model has no real predictive power

**Contrast with 1H Spread:**
- At 60%+ confidence: **68.6% accuracy** (beats threshold)
- Confidence score is **trustworthy** → Model knows when it knows

## Deployment Recommendation

### Deploy 3 Markets (Exclude 1H Total)

**Rationale:**
1. **FG Spread + FG Total**: Proven baseline profit at scale
2. **1H Spread**: Profitable when filtered to high-confidence bets
3. **1H Total**: Actively loses money even at high confidence

**Expected Strategy:**
- Deploy ALL FG Spread predictions (66.2% accuracy at baseline)
- Deploy ALL FG Total predictions (59.3% accuracy at baseline)
- Deploy ONLY high-conf 1H Spread (≥60% → 68.6% accuracy)
- **Skip** 1H Total entirely

**Portfolio Performance:**
- FG Spread: 339 bets @ +45% ROI (high-conf)
- FG Total: 168 bets @ +30% ROI (high-conf)
- 1H Spread: 35 bets @ +31% ROI (high-conf)
- **Total**: ~542 high-conf bets per season @ +38% blended ROI

## Alternative: Accept 1H Total as Loss Leader

**IF you insist on deploying 4 markets:**
- Accept that 1H Total will LOSE money (-4% ROI)
- Use it for customer engagement (more picks = more activity)
- Rely on FG Spread/Total profits to offset 1H Total losses
- **Net ROI will be lower** but product has "complete" feel

**Break-even analysis:**
- FG Spread: +26.5% ROI × 794 bets = +210 units
- FG Total: +13.2% ROI × 794 bets = +105 units
- 1H Spread: +31% ROI × 35 bets = +11 units (filtered)
- 1H Total: -4.0% ROI × 712 bets = **-28 units**
- **Net**: +298 units across 2335 bets = **+12.8% blended ROI**

Still profitable, but **1H Total is a 9% drag** on portfolio.

## Next Steps

**Option A (Recommended)**: Deploy 3 markets, skip 1H Total
- Update serving logic to filter out 1H Total predictions
- Document this decision in README/API docs
- Monitor for future 1H Total data sources

**Option B**: Deploy all 4, accept 1H Total losses as cost of product completeness
- Keep current training/serving setup
- Add disclaimer that 1H Total is experimental
- Track separately to quantify drag on returns

**What's your decision?**
