# Betting Splits Data Strategy

## Overview

This document outlines the strategy for handling betting splits data across backtesting and live prediction phases.

**Key Principle:** NO default/neutral values. Either use REAL data or EXCLUDE the feature entirely.

---

## Historical Splits Sources (Paid)

| Provider | URL | Cost | Data Range |
|----------|-----|------|------------|
| **Sports Insights** | sportsinsights.com/historical-betting-database | ~$500-1000/year | 2003-present |
| **Sportradar** | sportradar.com/betting-insights | Enterprise pricing | Full historical |
| **SportsDataIO** | sportsdata.io/historical-sports-data | ~$300-500/month | 2010-present |

### Sports Insights Dataset (Recommended)

The Sports Insights historical database includes:
- Opening and closing lines
- Final scores
- **Proprietary betting percentages** (public %, sharp %)
- Line movement data

Available for: NFL, MLB, NBA, NHL, WNBA, NCAAF, NCAAB

---

## Two-Track Model Architecture

### Track 1: Core Model (Backtesting Without Splits)

**Features Used:**
- Betting lines (spread, total, moneyline)
- Team statistics (PPG, PAPG, pace, ratings)
- Situational factors (rest, travel, B2B)
- Historical matchup data
- Line movement (open vs close)

**Features EXCLUDED (not defaulted):**
```python
# These features are NOT in the backtest model at all
# NOT set to default values - completely removed
EXCLUDED_FROM_BACKTEST = [
    "has_real_splits",
    "is_rlm_spread",
    "is_rlm_total",
    "spread_public_home_pct",
    "spread_public_away_pct",
    "spread_money_home_pct",
    "spread_money_away_pct",
    "over_public_pct",
    "under_public_pct",
    "sharp_side_spread",
    "sharp_side_total",
    "spread_ticket_money_diff",
    "total_ticket_money_diff",
]
```

**Training Configuration:**
```python
# In feature_config.py or models.py
BACKTEST_MODEL_FEATURES = {
    "core_features": ENABLED,      # Lines, stats, rest, H2H
    "splits_features": DISABLED,   # Completely excluded from training
}
```

### Track 2: Enhanced Model (Live Predictions With Splits)

**All Core Features PLUS:**
- Real-time betting splits from Action Network
- RLM signal detection
- Public % vs Money % divergence
- Sharp money indicators

**Live Pipeline:**
```python
async def get_live_features(game: dict) -> dict:
    # Get core features
    features = build_core_features(game)
    
    # Attempt to fetch real splits
    splits = await fetch_splits_action_network()
    
    if splits:
        # Add REAL splits features
        features.update(splits_to_features(splits))
        features["has_real_splits"] = 1
    else:
        # DO NOT USE THIS GAME FOR SPLITS-DEPENDENT PICKS
        # Only report core model prediction
        features["has_real_splits"] = 0
        # Splits features remain ABSENT (not set to defaults)
```

---

## Model Training Approach

### Option A: Single Core Model (Current Recommendation)

1. Train one model on core features only
2. Backtest validates core model accuracy
3. Live predictions ADD splits as post-model filter/boost

```
Core Model Prediction: PHO -4.5 (58% confidence)
                          │
                          ▼
Live Splits Check: 68% public on PHO, line moved to -5.0 (RLM detected)
                          │
                          ▼
Enhanced Prediction: PHO -4.5 AVOID (RLM against us)
                     -- or --
                     OKC +4.5 UPGRADE (sharp money signal)
```

### Option B: Dual Models (If Historical Splits Purchased)

1. Core Model: Trained without splits, backtested on all historical data
2. Enhanced Model: Trained WITH splits, backtested on Sports Insights dataset

```python
# Model selection in production
if has_real_splits:
    prediction = enhanced_model.predict(all_features)
else:
    prediction = core_model.predict(core_features)
```

---

## Backtesting Without Historical Splits

### What You CAN Validate:
- Edge from line analysis (opening vs closing)
- Statistical model accuracy
- Situational factors (rest, travel, etc.)
- Overall ROI without splits signals

### What You CANNOT Validate:
- RLM signal effectiveness
- Public % fade strategy
- Sharp money tracking accuracy

### Expected Impact:

| Scenario | Backtest Accuracy | Live Accuracy (est.) |
|----------|------------------|---------------------|
| Core model only | 55-58% | 55-58% |
| Core + live splits | 55-58% (backtest) | 57-61% (live) |
| Core + historical splits | 57-61% | 57-61% |

**Conclusion:** If core model backtests at 55%+, adding live splits should only IMPROVE performance. You're testing the baseline, and live splits are an enhancement.

---

## Implementation: Exclude Splits from Backtest

### Step 1: Update Feature Configuration

```python
# src/modeling/feature_config.py

FEATURE_GROUPS = {
    "core": {
        "enabled": True,
        "features": [
            "home_spread", "total_line", "rest_days_home", 
            "rest_days_away", "home_win_pct", "away_win_pct",
            # ... all core features
        ]
    },
    "splits": {
        "enabled": False,  # Disabled for backtest
        "features": [
            "has_real_splits", "is_rlm_spread", "is_rlm_total",
            "spread_public_home_pct", "spread_public_away_pct",
            # ... all splits features
        ]
    }
}
```

### Step 2: Modify Dataset Builder

```python
# scripts/build_complete_training_data.py

def build_training_data(include_splits: bool = False):
    """Build training data with optional splits features."""
    
    features = []
    
    # Always include core features
    features.extend(get_core_features(game))
    
    # Only include splits if explicitly requested AND available
    if include_splits:
        splits_data = get_historical_splits(game_date)  # From purchased dataset
        if splits_data:
            features.extend(splits_to_features(splits_data))
        else:
            # Skip this game - don't use defaults
            return None
    
    return features
```

### Step 3: Separate Live Prediction Pipeline

```python
# src/prediction/live.py

async def make_live_prediction(game: dict) -> dict:
    """Make prediction with real-time splits if available."""
    
    # Core prediction
    core_features = build_core_features(game)
    core_prediction = core_model.predict(core_features)
    
    # Attempt splits enhancement
    splits = await fetch_splits_action_network()
    
    result = {
        "prediction": core_prediction,
        "source": "core_model",
        "splits_available": False,
    }
    
    if splits:
        splits_features = splits_to_features(splits)
        
        # Use splits as filter/boost, not input to same model
        if splits_features["is_rlm_spread"] == 1:
            result["rlm_signal"] = detect_rlm_direction(splits)
            result["confidence_adjustment"] = calculate_rlm_boost(splits)
        
        result["splits_available"] = True
        result["splits_data"] = splits_features
    
    return result
```

---

## Decision Matrix: Buy Historical Splits?

| Factor | Buy Sports Insights | Use Live-Only Approach |
|--------|--------------------|-----------------------|
| Budget | Have $500-1000/year | Limited budget |
| Validation need | Must validate RLM strategy | Trust core model baseline |
| Model complexity | Want single unified model | Accept two-track approach |
| Time to launch | Can wait for data integration | Need to ship now |

### Recommendation:

**Start with live-only approach:**
1. Backtest core model (no splits)
2. Deploy with live Action Network splits
3. Track live performance for 1-2 months
4. If splits show clear edge, consider purchasing historical data for enhanced training

---

## Summary

| Phase | Splits Data | Features |
|-------|-------------|----------|
| **Backtesting** | None (excluded) | Core features only |
| **Live Predictions** | Real-time from Action Network | Core + splits when available |
| **Future (optional)** | Purchase Sports Insights | Full historical validation |

**Key Principle:** Never use default/neutral values for splits. Either use REAL data or EXCLUDE the feature entirely.
