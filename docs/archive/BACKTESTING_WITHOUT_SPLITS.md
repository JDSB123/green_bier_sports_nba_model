# Backtesting Strategy: Handling Betting Splits

## Overview

This document explains the proper approach for backtesting NBA prediction models when historical betting splits data is not available.

## The Problem

- **Betting splits** (public betting percentages, RLM signals) are **LIVE DATA ONLY**
- Action Network, SBRO, Covers, etc. provide current splits but NOT historical splits
- Historical backtesting cannot use real splits data
- Using placeholder/fake splits data would be misleading

## Current Data Availability

| Feature Type | Historical | Live |
|-------------|------------|------|
| **Betting Lines (spreads, totals, moneyline)** | 2008-2025 | Real-time |
| **Game Outcomes (scores)** | 2008-2025 | Real-time |
| **Team Statistics** | 2008-2025 | Real-time |
| **ELO Ratings** | 1946-2015 (538) | N/A |
| **Betting Splits (% public money)** | NOT AVAILABLE | Action Network |
| **RLM Signals (reverse line movement)** | NOT AVAILABLE | Derived from splits |

## The Solution: Two-Track Approach

### Track 1: Backtesting (Historical Validation)

**Features Used:**
- Betting lines (spreads, totals, moneyline)
- Team performance stats (PPG, PAPG, W-L, etc.)
- Rest days, travel, home/away
- Head-to-head history
- Historical line movements (if available)

**Features NOT Used:**
- `has_real_splits` = 0
- `is_rlm_spread` = 0 (default)
- `is_rlm_total` = 0 (default)
- `spread_public_home_pct` = 50 (default - neutral)
- All other splits features = defaults

**Backtest Accuracy Represents:**
- Core model performance on odds + stats
- Does NOT include RLM/splits signal boost
- This is the baseline that can be historically validated

### Track 2: Live Predictions (Production)

**Features Used:**
- ALL backtested features PLUS:
- `has_real_splits` = 1 (when fetched from Action Network)
- Real `is_rlm_spread`, `is_rlm_total`
- Real public betting percentages
- Real ticket vs money divergence

**Live Accuracy Represents:**
- Full model with RLM/splits enhancement
- Expected to outperform backtest (if splits signal is valuable)
- Cannot be historically validated

## Implementation

### 1. Model Training

Models are trained with splits features included but set to defaults for historical data:

```python
# In src/modeling/unified_features.py
BETTING_FEATURES = [
    Feature("has_real_splits", default=0.0),  # 0 = no real splits
    Feature("is_rlm_spread", default=0.0),     # 0 = no RLM detected
    Feature("spread_public_home_pct", default=50.0),  # 50% = neutral
    # ...
]
```

### 2. Backtest Execution

For proper historical backtesting, do NOT require `has_real_splits == 1`:

```python
# Correct approach: backtest on historical data without splits
# The model uses default values (neutral) for splits features
# This validates the CORE model (odds + stats)
```

### 3. Live Prediction

When making live predictions, fetch real splits:

```python
# In prediction pipeline
splits = await fetch_splits_action_network(date)
if splits:
    features["has_real_splits"] = 1
    features["is_rlm_spread"] = detect_rlm(splits)
    # ... populate real splits data
```

## Backtest Configuration

### For Historical Validation (Recommended)

```bash
# Run backtest WITHOUT requiring splits
python scripts/backtest_production.py \
    --start-date 2024-10-01 \
    --end-date 2025-01-01 \
    --allow-no-splits  # Use this flag if available
```

### What the Backtest Measures

| Metric | Includes Splits? | Validation Period |
|--------|-----------------|-------------------|
| Backtest Accuracy | NO | 2023-2025 |
| Backtest ROI | NO | 2023-2025 |
| Live Accuracy | YES | Ongoing |
| Live ROI | YES | Ongoing |

## Expected Performance

### Backtest (Without Splits)
- FG Spread: ~55-60% accuracy
- FG Total: ~55-60% accuracy
- 1H Spread: ~52-55% accuracy
- 1H Total: ~52-55% accuracy

### Live (With Splits) - Theoretical Enhancement
- RLM signal adds +2-5% edge when triggered
- Not all games have RLM signals (maybe 10-20%)
- Net impact on overall accuracy: +0.5-1%

## Key Principles

1. **NO FAKE DATA**: Never use placeholder splits as if they were real
2. **SEPARATE VALIDATION**: Backtest accuracy is without splits; live is with
3. **TRANSPARENT REPORTING**: Always disclose which mode a metric comes from
4. **DEFAULT VALUES ARE NEUTRAL**: 50% splits = no signal = no impact

## Moneyline Integration

Since moneyline (h2h) data IS available historically:
- Kaggle: 2008-2023 (full coverage until mid-2023)
- The Odds API: 2023-2025 (full coverage)

Moneyline CAN be used in backtesting and adds to core model accuracy.

## Summary

| Aspect | Backtest | Live |
|--------|----------|------|
| Splits Features | Defaults (neutral) | Real data |
| RLM Signals | No | Yes |
| Historical Validation | Yes | No |
| Model Validated | Core (odds+stats) | Core + RLM |
| Accuracy Comparison | Baseline | Enhanced |

This approach ensures:
- Honest historical validation
- No data leakage or placeholder contamination
- Clear separation between validated and enhanced performance
