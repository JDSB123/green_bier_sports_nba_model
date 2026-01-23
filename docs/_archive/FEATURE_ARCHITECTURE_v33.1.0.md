# Feature Architecture - NBA Model v33.1.0

**Last Updated:** 2026-01-19
**Version:** v33.1.0

---

## 🏗️ ARCHITECTURE OVERVIEW

### Unified Feature Schema
All models (1H + FG, Spread + Total) use the **SAME feature names** for consistency:
- Training: Models trained on unified feature set
- Prediction: Period-specific **values** with unified **names**

```python
# Example: PPG feature
FG Model: "home_ppg" = 115.0  (full game average)
1H Model: "home_ppg" = 57.5   (first half average) ← different VALUE, same NAME
```

---

## 📊 FEATURE CATEGORIES

### 1. Period-Specific Features (Different for 1H vs FG)

These features have DIFFERENT values for 1H vs FG:

| Category | Features | 1H Source | FG Source |
|----------|----------|-----------|-----------|
| **Scoring** | `home_ppg`, `away_ppg`, `home_papg`, `away_papg`, `ppg_diff` | 1H game data (Q1+Q2) | Full game data |
| **Margins** | `home_margin`, `away_margin` | 1H margin stats | FG margin stats |
| **Win Rates** | `home_win_pct`, `away_win_pct` | 1H win% | FG win% |
| **Pace** | `home_pace`, `away_pace`, `expected_pace` | 1H pace | FG pace |
| **Form** | `home_l5_margin`, `away_l5_margin`, `home_l10_margin`, `away_l10_margin` | 1H L5/L10 | FG L5/L10 |
| **Consistency** | `home_margin_std`, `away_margin_std`, `home_score_std`, `away_score_std` | 1H volatility | FG volatility |
| **Efficiency** | `home_ortg`, `home_drtg`, `home_net_rtg`, `away_ortg`, `away_drtg`, `away_net_rtg` | 1H ORtg/DRtg | FG ORtg/DRtg |
| **Predictions** | `predicted_margin`, `predicted_total` | 1H prediction | FG prediction |
| **H2H** | `h2h_margin` | 1H H2H | FG H2H |

**Implementation:**
```python
# In features.py
if predicting_1h:
    features["home_ppg"] = home_1h_stats["ppg_1h"]  # 1H data
else:
    features["home_ppg"] = home_fg_stats["ppg"]     # FG data
```

---

### 2. Shared Features (Same for 1H and FG)

These features are **game-level** and identical for both periods:

| Category | Features | Notes |
|----------|----------|-------|
| **Rest** | `home_rest`, `away_rest`, `rest_diff`, `home_b2b`, `away_b2b`, `home_rest_adj`, `away_rest_adj`, `rest_margin_adj` | Days since last game (same for 1H and FG) |
| **Travel** | `away_travel_distance`, `away_timezone_change`, `away_travel_fatigue`, `is_away_long_trip`, `is_away_cross_country`, `away_b2b_travel_penalty`, `travel_advantage` | Travel metrics (game-level) |
| **Injuries** | `has_injury_data`, `home_injury_impact_ppg`, `away_injury_impact_ppg`, `injury_margin_adj`, `home_star_out`, `away_star_out`, `home_injury_spread_impact`, `away_injury_spread_impact` | Injury status at game time |
| **Home Court** | `dynamic_hca`, `home_court_advantage` | Team-specific HCA (~3 pts for FG, ~1.5 for 1H) |
| **Elo** | `home_elo`, `away_elo`, `elo_diff`, `elo_prob_home` | FiveThirtyEight Elo ratings |
| **Betting** | `has_real_splits`, `public_home_pct`, `sharp_money_side`, `rlm_indicator`, `public_split_extreme`, `consensus_spread`, `consensus_total` | Betting market data |
| **Strength of Schedule** | `home_sos_rating`, `away_sos_rating`, `sos_diff`, `home_recent_sos`, `away_recent_sos` | Opponent quality |
| **ATS Performance** | `home_ats_pct`, `away_ats_pct`, `home_over_pct`, `away_over_pct` | Against-the-spread history |

**Implementation:**
```python
# These are computed once and used for both periods
features["home_rest"] = (game_date - last_game_date).days
features["home_elo"] = fetch_elo(team, game_date)
# No period suffix needed
```

---

## 🔄 FEATURE MAPPING (1H Models)

### The Challenge
1H models were **trained** on unified feature names (`home_ppg`, not `home_ppg_1h`)
But at **prediction time**, we need to provide 1H-specific data.

### The Solution: Runtime Mapping
Function: `map_1h_features_to_fg_names()` in [src/prediction/engine.py](../src/prediction/engine.py#L86-L180)

**How It Works:**
```python
# Feature engineering creates BOTH versions:
features = {
    "home_ppg": 115.0,      # FG average
    "home_ppg_1h": 57.5,    # 1H average
    # ... more features
}

# For 1H prediction, map 1H values to FG names:
if predicting_1h:
    mapped = map_1h_features_to_fg_names(features)
    # Result:
    # {
    #     "home_ppg": 57.5,  # ← CHANGED: now has 1H value
    #     "home_rest": 2.0,  # ← UNCHANGED: shared feature
    #     # ...
    # }
```

### Mapped Features (Period-Specific Only)
```python
PERIOD_SPECIFIC_MAPPINGS = {
    "home_ppg_1h": "home_ppg",
    "away_ppg_1h": "away_ppg",
    "home_margin_1h": "home_margin",
    "predicted_margin_1h": "predicted_margin",
    "predicted_total_1h": "predicted_total",
    # ... see engine.py for complete list
}
```

### Shared Features (No Mapping)
```python
# These are NOT mapped (same for 1H and FG):
- home_rest, away_rest, rest_diff
- away_travel_distance, away_travel_fatigue
- home_elo, away_elo, elo_diff
- home_injury_impact_ppg, dynamic_hca
# ... etc
```

---

## 🎯 ADVANCED STATS & RATING SYSTEMS

### 1. Elo Ratings (FiveThirtyEight)
**Source:** [src/ingestion/github_data.py](../src/ingestion/github_data.py)
**Features:**
- `home_elo`: Team's current Elo rating (~1500 avg)
- `away_elo`: Opponent's Elo rating
- `elo_diff`: Home - Away Elo
- `elo_prob_home`: Elo-implied win probability

**Usage:**
```python
from src.ingestion.github_data import fetch_fivethirtyeight_elo
elo_df = await fetch_fivethirtyeight_elo()
```

**Status:** ✅ Implemented and available

---

### 2. Adjusted Efficiency Ratings
**Computed in:** [src/modeling/features.py](../src/modeling/features.py#L166-L178)
**Features:**
- `home_ortg`, `away_ortg`: Offensive Rating (points per 100 possessions)
- `home_drtg`, `away_drtg`: Defensive Rating (points allowed per 100)
- `home_net_rtg`, `away_net_rtg`: Net Rating (ORtg - DRtg)
- `net_rating_diff`: Home NetRtg - Away NetRtg

**Adjustment:**
```python
# Opponent-adjusted ratings
adj_ortg = ppg * (league_avg / opponent_strength)
adj_drtg = papg * (opponent_strength / league_avg)
net_rtg = adj_ortg - adj_drtg
```

**Status:** ✅ Implemented (basic version)

---

### 3. Strength of Schedule (SOS)
**Computed in:** [src/modeling/features.py](../src/modeling/features.py#L764-L842)
**Features:**
- `home_sos_rating`: Average opponent win%
- `away_sos_rating`: Average opponent win%
- `sos_diff`: Home - Away SOS
- `home_recent_sos`: Last 5 games SOS
- `away_recent_sos`: Last 5 games SOS

**Calculation:**
```python
sos_rating = mean(opponent_win_percentages)
```

**Status:** ✅ Implemented

---

### 4. Against-The-Spread (ATS) Performance
**Features:**
- `home_ats_pct`: Home team's ATS win%
- `away_ats_pct`: Away team's ATS win%
- `home_over_pct`: Home team's over%
- `away_over_pct`: Away team's over%

**Status:** ⚠️ Defined in schema, needs data source verification

---

### 5. KenPom / Bart Torvik Style Ratings
**Current Implementation:** Basic efficiency ratings (ORtg/DRtg/NetRtg)

**Potential Enhancements:**
- Tempo-free stats (per 100 possessions) ✅ Already implemented
- Four Factors (eFG%, TOV%, ORB%, FT Rate) ❌ Not implemented
- Opponent-adjusted ratings ✅ Basic adjustment implemented
- Luck-adjusted ratings ❌ Not implemented

**To Add Full KenPom/Torvik Style:**
```python
# Four Factors (would require box score data)
features["home_efg_pct"] = (FGM + 0.5 * 3PM) / FGA
features["home_tov_pct"] = TOV / (FGA + 0.44 * FTA + TOV)
features["home_orb_pct"] = ORB / (ORB + opp_DRB)
features["home_ft_rate"] = FTA / FGA

# Pythagorean Win% (luck-adjusted)
features["home_pyth_wins"] = ppg^10.25 / (ppg^10.25 + papg^10.25)
```

**Status:** ⚠️ Partial (basic efficiency only)

---

## 📋 COMPLETE FEATURE LIST

### Total Features: ~118 features

**Core Stats (15):**
- PPG, PAPG, margin, win%, pace, predicted margin/total, differentials

**Efficiency (7):**
- ORtg, DRtg, NetRtg for home/away + diffs

**Form (10):**
- L5/L10 margins, std deviations, form trends

**Rest & Schedule (8):**
- Rest days, B2B, rest adjustments

**Travel (7):**
- Distance, timezone, fatigue, trip type

**Injuries (8):**
- Impact PPG, spread impact, star out flags

**Home Court (2):**
- Dynamic HCA, team-specific HCA

**Elo (4):**
- Home/Away Elo, diff, win probability

**Betting (12):**
- Public splits, RLM, sharp money, consensus lines

**Strength of Schedule (5):**
- SOS ratings, recent SOS

**ATS Performance (4):**
- ATS%, Over%

**Market Lines (4):**
- Spread/total lines, vs predicted

**H2H (3):**
- H2H margin, games, win%

**Standings (3):**
- Position, position diff

**Misc (26):**
- Various derived features and interactions

---

## ✅ PREDICTION READINESS CHECKLIST

### Required for ALL Predictions (FG + 1H)
- [x] Core team stats (PPG, PAPG, margin)
- [x] Efficiency ratings (ORtg, DRtg, NetRtg)
- [x] Rest days and B2B flags
- [x] Travel metrics (distance, timezone)
- [x] Home court advantage
- [x] Predicted margin/total (from formulas)
- [x] Form (L5/L10 margins)
- [x] Consistency (std deviations)

### Required for 1H Predictions ONLY
- [x] 1H-specific scoring stats (ppg_1h, margin_1h)
- [x] 1H efficiency (ortg_1h, drtg_1h)
- [x] 1H pace and form
- [x] Predicted margin_1h and total_1h
- [x] Feature mapping function

### Optional (Enhance Accuracy)
- [x] Elo ratings (FiveThirtyEight)
- [x] Strength of schedule
- [ ] Public betting splits (requires Action Network login)
- [ ] Injury data (requires ESPN/API scraping)
- [x] ATS performance (schema defined, needs data)

### Advanced (Future Enhancement)
- [ ] Four Factors (eFG%, TOV%, ORB%, FT rate)
- [ ] Pythagorean win% (luck-adjusted)
- [ ] Player-level impact metrics
- [ ] Lineup-specific stats

---

## 🚀 USAGE EXAMPLES

### Example 1: FG Prediction
```python
from src.features import RichFeatureBuilder
from src.prediction import UnifiedPredictionEngine

# Build features (FG data)
builder = RichFeatureBuilder(season="2025-2026")
features = builder.build_game_features(
    historical_df=historical_data,
    home_team="Los Angeles Lakers",
    away_team="Boston Celtics",
    game_date="2026-01-20",
    fg_spread=-5.5,
    fg_total=225.0,
)

# Predict FG markets
engine = UnifiedPredictionEngine(models_dir="models/production")
predictions = engine.predict_full_game(
    features=features,
    spread_line=-5.5,
    total_line=225.0,
)

print(predictions["spread"]["bet_side"])    # "home" or "away"
print(predictions["spread"]["confidence"])  # 0.65
print(predictions["total"]["bet_side"])     # "over" or "under"
```

### Example 2: 1H Prediction
```python
# Build features (includes 1H data)
features = builder.build_game_features(
    historical_df=historical_data,
    home_team="Los Angeles Lakers",
    away_team="Boston Celtics",
    game_date="2026-01-20",
    fh_spread=-2.5,  # 1H spread
    fh_total=112.0,  # 1H total
)

# Predict 1H markets
predictions = engine.predict_first_half(
    features=features,
    spread_line=-2.5,
    total_line=112.0,
)

# Features automatically mapped via map_1h_features_to_fg_names()
print(predictions["spread"]["bet_side"])
```

### Example 3: Check Feature Completeness
```python
from src.modeling.unified_features import get_feature_defaults, UNIFIED_FEATURE_NAMES

# Get all required features with defaults
defaults = get_feature_defaults()
print(f"Total features: {len(defaults)}")

# Check if your features dict is complete
missing = set(UNIFIED_FEATURE_NAMES) - set(features.keys())
if missing:
    print(f"Missing features: {missing}")
```

---

## 🔧 TROUBLESHOOTING

### Issue: "MISSING REQUIRED FEATURES" Error
**Cause:** Feature engineering didn't create all required features

**Fix:**
```python
# Set feature validation mode to warn (for debugging)
import os
os.environ["PREDICTION_FEATURE_MODE"] = "warn"

# Or get defaults for missing features
from src.modeling.unified_features import get_feature_defaults
features.update(get_feature_defaults())
```

### Issue: 1H Predictions Failing
**Cause:** Missing 1H quarter data (Q1, Q2 scores)

**Fix:** Ensure historical_df has `home_q1`, `home_q2`, `away_q1`, `away_q2` columns
```python
# v33.1.0: Graceful degradation
# If 1H data missing → skips 1H markets, FG markets still work
```

### Issue: Feature Values Look Wrong
**Cause:** 1H feature mapping applied to FG prediction (or vice versa)

**Fix:** Mapping only happens inside `PeriodPredictor.predict_spread/total` for 1H period
```python
# Automatic in v33.1.0:
if self.period == "1h":
    features = map_1h_features_to_fg_names(features)  # Only for 1H
```

---

## 📚 REFERENCES

- **Feature Definitions:** [src/modeling/unified_features.py](../src/modeling/unified_features.py)
- **Feature Engineering:** [src/modeling/features.py](../src/modeling/features.py)
- **Feature Mapping:** [src/prediction/engine.py](../src/prediction/engine.py#L86-L180)
- **Elo Data:** [src/ingestion/github_data.py](../src/ingestion/github_data.py)
- **Validation:** [src/prediction/feature_validation.py](../src/prediction/feature_validation.py)

---

**Version:** v33.1.0
**Last Updated:** 2026-01-19
**Status:** Production Ready ✅
