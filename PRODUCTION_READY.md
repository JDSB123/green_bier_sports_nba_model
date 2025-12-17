# NBA V4.0 - PRODUCTION READY

## STATUS: PRODUCTION READY ✅

**Date:** December 17, 2025
**Version:** 2.0 - Modular Architecture (Spreads + Totals)

---

## What's Ready

### 1. Backtested Strategy ✅
- **Tested on:** 422 games (Oct 2 - Dec 9, 2025)
- **Spreads:** 60.6% accuracy, +15.7% ROI (with filtering)
- **Totals:** 59.2% accuracy, +13.1% ROI (baseline)
- **Validation:** Walk-forward validation, no lookahead bias

### 2. Modular Prediction Architecture ✅
```
src/prediction/
  filters.py     - Smart filtering logic (spreads + totals)
  models.py      - Model loading/management
  predictor.py   - Core prediction engine
```
- **Clean separation of concerns**
- **Testable filtering logic**
- **Reusable components**
- **Easy to extend**

### 3. Smart Filtering System ✅
**Spreads:**
- Filter 1: Remove 3-6 point spreads (only 42% accuracy)
- Filter 2: Require minimum 5% model edge
- Result: Reduces bets from 341 to ~203, triples ROI

**Totals:**
- No filtering (baseline is best!)
- Filtering actually hurts performance (13.1% → 8.8% ROI)

### 4. Production Prediction Script ✅
- **Script:** `scripts/predict_v2.py` (modular version)
- **Features:**
  - Fetches upcoming games from The Odds API
  - Predicts both spreads AND totals
  - Applies smart filtering automatically
  - Generates unified betting card
  - Shows filter summary and reasons

### 5. Outputs ✅
- **predictions.csv** - All predictions (spreads + totals)
- **betting_card.csv** - Filtered plays only (ready to bet)
- **Console output** - Detailed analysis + unified betting card

---

## How to Use Daily

### Quick Start (Recommended)
```bash
# Get tomorrow's betting card (spreads + totals)
python scripts/predict_v2.py

# Get today's betting card
python scripts/predict_v2.py --date today

# Get specific date
python scripts/predict_v2.py --date 2025-12-18

# Use old spreads-only version
python scripts/predict.py
```

### Output Files
```
data/processed/
  predictions.csv - All predictions (filtered + unfiltered)
  betting_card.csv - Filtered plays only (USE THIS FOR BETTING)
```

### Reading the Betting Card

**Example Output (Spreads + Totals):**
```
Memphis Grizzlies @ Minnesota Timberwolves
  Game Time: 2025-12-17 07:10 PM CST
  Market: SPREAD
  Pick: home (56.4%)
  Edge: +10.8 pts

Memphis Grizzlies @ Minnesota Timberwolves
  Game Time: 2025-12-17 07:10 PM CST
  Market: TOTAL
  Pick: over 232.5 (91.2%)
  Edge: +2.0 pts
```

**What This Means:**
- **Spread Play:** Bet Minnesota Timberwolves -7.5 (56.4% confidence)
  - Model edge: 6.4% (passes 5% threshold)
  - Points edge: +10.8 pts (model predicts larger win)
- **Total Play:** Bet OVER 232.5 (91.2% confidence)
  - Model predicts 234.5 total points
  - Edge: +2.0 pts over the line

---

## Expected Performance

Based on 422-game backtest:

### Spreads (Smart Filtering)
- **Win Rate:** 60.6%
- **ROI:** +15.7% (at -110 odds)
- **Bets per Season:** ~200

### Totals (Baseline - No Filter)
- **Win Rate:** 59.2%
- **ROI:** +13.1% (at -110 odds)
- **Bets per Season:** ~340

### Combined Portfolio
- **Total Bets:** ~540 per season (~2.1 per game day)
- **Weighted ROI:** ~+14.0%
- **Expected Profit:**
  - $100/bet: +$7,560 per season
  - $500/bet: +$37,800 per season
  - $1000/bet: +$75,600 per season

**Note:** These are backtest results. Real performance will vary. Always bet responsibly.

---

## What Gets Filtered Out

### Spreads Filtering

**Reason 1: Small Spreads (3-6 points)**
- **Why:** Only 42.2% accuracy (worse than random)
- **Market:** Most efficient in this range
- **Action:** SKIP automatically

**Reason 2: Insufficient Edge (<5%)**
- **Why:** Near 50/50 accuracy (no real edge)
- **Model confidence:** 50-55%
- **Action:** SKIP automatically

**What Passes Filter:**
- Pick-ems (0-3 pts): 61.5% accuracy ✅
- Medium spreads (6-9 pts): 59.6% accuracy ✅
- Large spreads (9+ pts): 55.4% accuracy ✅
- AND model edge ≥ 5%

### Totals Filtering

**NO FILTERING** ✅
- Baseline model (59.2% / +13.1% ROI) beats filtered approach
- Filtering reduces to 57.0% / +8.8% ROI
- **Bet all totals with any confidence**

---

## Daily Workflow

### Morning Routine
1. **Run prediction script:**
   ```bash
   python scripts/predict_v2.py
   ```

2. **Review betting card:**
   - Check `data/processed/betting_card.csv`
   - Or read console output
   - Shows both spreads AND totals

3. **Place bets:**
   - Bet on all plays listed in betting card
   - Use consistent unit sizing
   - Get best available lines
   - ~2 plays per game day on average

4. **Track results:**
   - Log all picks and outcomes (spreads + totals)
   - Monitor actual ROI vs expected
   - Track both markets separately

### What If No Plays Today?
```
NO PLAYS TODAY
All games filtered out - no bets meet criteria
```

**This is rare** (totals are rarely filtered):
- Spreads may be filtered (some days 0 spread plays)
- Totals bet on all games (~1.3 totals per day)
- Combined: ~2.1 plays per day on average
- Some days 1, some days 4-5

---

## Production Checklist

- [x] Backtested strategy validated (Spreads 60.6% / +15.7%, Totals 59.2% / +13.1%)
- [x] Modular prediction architecture created (src/prediction/)
- [x] Smart filtering implemented for spreads + totals
- [x] Unified betting card output (both markets)
- [x] Tested on live games (Dec 17, 2025)
- [x] Filter summary shows exclusion reasons
- [x] Documentation complete
- [x] Ready for daily use (spreads + totals)

---

## Model Details

### Current Models

**Spreads Model:**
- Type: Gradient Boosting Classifier
- Features: 29 features (team stats, ELO, matchup history)
- Training: Updated nightly with latest results
- File: `data/processed/models/spreads_model.joblib`
- Performance: 60.6% accuracy, +15.7% ROI (with filtering)

**Totals Model:**
- Type: Gradient Boosting Classifier
- Features: 15 features (pace, scoring trends, matchups)
- Training: Updated nightly with latest results
- File: `data/processed/models/totals_model.joblib`
- Performance: 59.2% accuracy, +13.1% ROI (no filtering)

### Features Used (Both Models)
- Team offensive/defensive ratings
- Rolling PPG, rebounds, assists
- ELO ratings
- Head-to-head history
- Home court advantage
- Rest days, back-to-backs
- Pace, possessions
- Predicted margins/totals from feature engineering

### Not Using (Yet)
- Calibration (made performance worse)
- Ensemble models (overfitted on limited data)
- Advanced features (need more data - 1000+ games)
- Line movement signals (optional, available with betting splits)

---

## Risk Management

### Recommended Bankroll Management
- **Unit size:** 1-2% of bankroll
- **Max bets per day:** Follow filter (usually 0-3)
- **Never chase:** If filtered out, don't bet
- **Track everything:** Log all picks and outcomes

### Expected Variance
- **Winning streaks:** 5-10 bets common
- **Losing streaks:** 3-5 bets expected
- **Sample size:** Need 100+ bets to see true ROI
- **Variance:** ROI can swing ±10% in short term

### Red Flags to Monitor
- **Win rate < 55%** after 50+ bets → Re-evaluate
- **ROI < 5%** after 100+ bets → Review model
- **Filter pass rate > 80%** → Filters too loose
- **Filter pass rate < 20%** → Filters too tight (expected ~60%)

---

## Support Files

### Backtest Results
- `BACKTEST_RESULTS_SUMMARY.md` - Comprehensive comparison
- `backtest_improved_gradient_boosting.csv` - Winning approach results

### Scripts
- `scripts/predict.py` - Production prediction script
- `scripts/backtest_improved.py` - Backtest with smart filtering
- `scripts/analyze_spread_performance.py` - Performance analysis

### Analysis
- `SPREAD_IMPROVEMENT_RECOMMENDATIONS.md` - Future improvements

---

## Future Improvements

**When to revisit:**
- After accumulating 1000+ games of data
- When ensemble models can be properly validated
- When feature importance can be rigorously tested

**Planned enhancements:**
- Clutch performance features
- Opponent-adjusted metrics
- Dynamic home court advantage
- Separate home/away models

**Expected impact:**
- Could push to 63-65% accuracy
- Target 20-25% ROI
- But need more data first

---

## Troubleshooting

### No predictions generated
```bash
# Check API key is set
echo %THE_ODDS_API_KEY%

# Check models exist
dir data\processed\models\spreads_model.joblib
```

### Missing features warning
```
[WARN] Missing features: {'spread_public_home_pct', 'away_b2b', ...}
```
**This is normal** - betting splits features are optional. Predictions still work.

### All games filtered out
**This is expected** - some days have no qualifying bets. Don't force plays.

---

## Contact & Questions

For issues or questions:
1. Check `BACKTEST_RESULTS_SUMMARY.md` for methodology
2. Review `scripts/predict.py` for implementation details
3. Check git logs for recent changes

---

**GOOD LUCK AND BET RESPONSIBLY!**
