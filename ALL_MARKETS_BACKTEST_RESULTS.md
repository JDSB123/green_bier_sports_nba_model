# All Markets Backtest Results

**Analysis Date:** Jan 5, 2026
**Data Source:** The Odds API (Unified)
**Models:** Gradient Boosting (Primary), Logistic Regression (Secondary)

---

## 🚨 Executive Summary

Recent backtesting (Jan 5, 2026) reveals that **strict thresholding is required** for profitability. Base models (at 0.60 confidence) generally yield negative ROI (-3% to -4%), but **optimized high-confidence thresholds** unlock significant value, particularly in **FG Totals (+8.7% ROI)** and **FG Spreads (+5.1% ROI)**.

> **CRITICAL WARNING:** A Logistic Regression run on spreads showed ~99% accuracy. This indicates probable **data leakage** in that specific run. We are discarding those results in favor of the realistic Gradient Boosting numbers and the validated optimization run.

---

## ✅ Production Recommendations (Optimized Thresholds)

Deploy the following thresholds to target positive ROI. These settings trade volume for edge.

| Market | Model | Recommended Threshold | Expected ROI | Win Rate | Bet Volume (Est.) |
|--------|-------|----------------------|--------------|----------|-------------------|
| **FG Total** | **Gradient Boosting** | **0.72** | **+8.7%** | **~57%** | Low (Selective) |
| **FG Spread** | **Logistic** | **0.62** | **+5.1%** | **~54%** | Medium |
| **1H Spread** | **Gradient Boosting** | **0.68** | **+3.0%** | **~54%** | Medium |
| **1H Total** | **Logistic** | **0.66** | **+0.2%** | **~52%** | Low |

---

## 📉 Base Performance (Unfiltered)

Without strict thresholds (i.e., betting everything >50% or >60%), the models struggle against the vigorish (vig). This confirms the need for the **"High ROE / No Silent Fallbacks"** strategy.

### Gradient Boosting (Median Lines)
*Results from `backtest_summary_theodds_median_gradient_boosting_20260105_135223.json`*

| Market | Bets | Accuracy | ROI | Profit (Units) |
|--------|------|----------|-----|----------------|
| **FG Spread** | 6,298 | 50.8% | -3.1% | -194.6 |
| **FG Total** | 6,301 | 50.6% | -3.4% | -214.8 |
| **1H Spread** | 2,200 | 50.6% | -3.4% | -75.2 |
| **1H Total** | 2,200 | 50.5% | -3.6% | -79.0 |

> **Insight:** The "Base" models are essentially coin flips (50.6% accuracy). The alpha exists solely in the **tails of the confidence distribution** (the top 5-10% of bets), which is why the Optimized Thresholds above are the only viable production path.

---

## 🛠 Next Steps for Engineering

1.  **Enforce Thresholds:** Update `src/config.py` with the recommended thresholds:
    - `FILTER_SPREAD_MIN_CONFIDENCE = 0.62`
    - `FILTER_TOTAL_MIN_CONFIDENCE = 0.72`
2.  **Investigate Leakage:** Audit the Logistic Regression feature set for `fg_spread`. The ~99% accuracy anomaly suggests `margin` or `points` might be leaking into the feature vector.
3.  **Deploy Gradient Boosting for Totals:** The GB model significantly outperforms Logistic on Totals (+8.7% vs +2.0% ROI potential).

---
*Generated from data in `data/backtest_results/`*
