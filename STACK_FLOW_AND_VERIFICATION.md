# Complete Stack Flow & Verification Guide

## 🔄 Complete Stack Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKER CONTAINER START                       │
│              docker-entrypoint-backtest.sh                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 0: VALIDATION                                             │
│  ├─ Validate API keys (API_BASKETBALL_KEY, THE_ODDS_API_KEY)   │
│  ├─ Validate Python environment                                 │
│  └─ Check critical imports (models, features, ingestion)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: FETCH GAME OUTCOMES                                    │
│  scripts/build_fresh_training_data.py                           │
│  └─ FreshDataPipeline.fetch_game_outcomes()                     │
│     ├─ For each season:                                         │
│     │  ├─ APIBasketballClient.ingest_essential()                │
│     │  │  ├─ fetch_teams() → /teams endpoint                    │
│     │  │  ├─ fetch_games() → /games endpoint                    │
│     │  │  ├─ fetch_statistics() → /statistics endpoint          │
│     │  │  └─ fetch_game_stats_teams() → /games/statistics/teams │
│     │  └─ Process games with Q1-Q4 scores                       │
│     └─ Output: outcomes_df (games with scores)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: FETCH BETTING LINES                                    │
│  FreshDataPipeline.fetch_betting_lines()                        │
│  ├─ fetch_participants() → /sports/.../participants             │
│  ├─ Try historical odds first (if paid plan)                    │
│  │  ├─ fetch_historical_odds() → /historical/sports/.../odds    │
│  │  └─ For each event:                                          │
│  │     └─ fetch_event_odds() → /events/{id}/odds (1H/Q1)        │
│  └─ Fallback to current odds:                                   │
│     ├─ fetch_events() → /sports/.../events                      │
│     ├─ fetch_odds() → /sports/.../odds (FG markets)             │
│     └─ For each event:                                          │
│        └─ fetch_event_odds() → /events/{id}/odds (1H/Q1)        │
│  └─ Output: lines_df (spread/total lines)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: MERGE OUTCOMES + LINES                                 │
│  FreshDataPipeline.merge_outcomes_and_lines()                   │
│  └─ Match games to betting lines by team names + date           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: COMPUTE LABELS                                         │
│  FreshDataPipeline.compute_labels()                             │
│  ├─ spread_covered = (actual_margin > -spread_line)             │
│  ├─ total_over = (actual_total > total_line)                    │
│  ├─ 1h_spread_covered = (actual_1h_margin > -1h_spread_line)   │
│  └─ 1h_total_over = (actual_1h_total > 1h_total_line)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4.5: ENRICH WITH BETTING SPLITS                           │
│  FreshDataPipeline.enrich_with_betting_splits()                 │
│  ├─ Try fetch_betting_splits() → /sports/.../betting-splits     │
│  └─ Fallback to Action Network (if credentials available)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: VALIDATE DATASET                                       │
│  FreshDataPipeline.validate_dataset()                           │
│  ├─ Check required columns exist                                │
│  ├─ Check data quality (null percentages)                       │
│  └─ Report coverage (spread lines, total lines, etc.)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: SAVE TRAINING DATA                                     │
│  └─ Save to: data/processed/training_data.csv                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: RUN BACKTEST                                           │
│  scripts/backtest.py                                            │
│  ├─ load_training_data() → Load training_data.csv               │
│  ├─ For each market (fg_spread, fg_total, etc.):                │
│  │  ├─ Walk-forward validation (train on past, predict next)    │
│  │  ├─ FeatureEngineer.build_game_features()                    │
│  │  │  ├─ compute_team_rolling_stats() → Historical stats       │
│  │  │  ├─ compute_rest_days() → Rest calculation                │
│  │  │  ├─ compute_travel_features() → Travel fatigue            │
│  │  │  ├─ compute_dynamic_hca() → Home court advantage          │
│  │  │  ├─ compute_h2h_stats() → Head-to-head                    │
│  │  │  └─ compute_sos_features() → Strength of schedule         │
│  │  ├─ Model.fit() → Train model on historical data             │
│  │  ├─ Model.predict_proba() → Get probabilities                │
│  │  └─ Calculate accuracy/ROI                                   │
│  └─ Save results to: data/processed/all_markets_backtest_results.csv │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT: Backtest Results                                       │
│  ├─ all_markets_backtest_results.csv                            │
│  └─ ALL_MARKETS_BACKTEST_RESULTS.md                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ Verification Steps

### Step 1: Verify Container Entry Point

**Test:**
```bash
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
# Inside container:
python -c "from src.modeling.models import SpreadsModel, TotalsModel; print('OK')"
```

**Expected Output:**
```
✓ src.modeling.models imported successfully
OK
```

**What to check:**
- ✅ No import errors
- ✅ All critical modules load

---

### Step 2: Verify Data Fetching

**Test:**
```bash
docker compose -f docker-compose.backtest.yml run --rm backtest-data
```

**Expected Output:**
```
============================================================
STEP 1: FETCHING FRESH DATA
============================================================
Seasons: 2024-2025,2025-2026
...
✓ Fetched 1200+ game outcomes
✓ Fetched 800+ betting line records
✓ Matched 750+/1200+ games with betting lines
✓ Enriched 200+/1200+ games with betting splits
✓ Training data saved to /app/data/processed/training_data.csv
```

**What to check:**
- ✅ Games fetched from API-Basketball
- ✅ Betting lines fetched from The Odds API
- ✅ Training data file created
- ✅ Check file: `data/processed/training_data.csv`

**Verify file exists:**
```bash
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
ls -lh /app/data/processed/training_data.csv
wc -l /app/data/processed/training_data.csv
```

---

### Step 3: Verify API Endpoints Are Used

**Test:** Check logs for endpoint calls

**Look for in logs:**
```
Fetching participants reference from The Odds API...
✓ Fetched 30 participants for team validation

Fetching game outcomes for seasons: ['2024-2025', '2025-2026']
  Fetching all essential endpoints for season 2024-2025...
Ingesting: teams
  [OK] 30 records -> /app/data/raw/api_basketball/teams_*.json
Ingesting: games
  [OK] 600+ records -> /app/data/raw/api_basketball/games_*.json
Ingesting: statistics
  [OK] 30 records -> /app/data/raw/api_basketball/statistics_*.json
Ingesting: game_stats_teams
  [OK] 500+ records -> /app/data/raw/api_basketball/game_stats_teams_*.json

Fetching betting lines for 180 unique dates...
  Processing date 1/180...
  ✓ Historical odds endpoint available
  ...
  ✓ Fetched event odds for event abc123 (1H/Q1 markets)
  ...
```

**What to check:**
- ✅ `ingest_essential()` called (not just `fetch_games()`)
- ✅ Participants endpoint called
- ✅ Event-specific odds called for 1H/Q1 markets
- ✅ Betting splits endpoint attempted

---

### Step 4: Verify Feature Engineering

**Test:**
```bash
docker compose -f docker-compose.backtest.yml run --rm backtest-shell
python -c "
from src.modeling.features import FeatureEngineer
import pandas as pd
from datetime import datetime

fe = FeatureEngineer()
game = pd.Series({
    'home_team': 'LAL',
    'away_team': 'BOS',
    'date': datetime(2025, 12, 18),
    'spread_line': -5.0,
    'total_line': 220.0
})

historical = pd.DataFrame([
    {'date': datetime(2025, 12, 15), 'home_team': 'LAL', 'away_team': 'MIA', 'home_score': 110, 'away_score': 105},
    {'date': datetime(2025, 12, 16), 'home_team': 'BOS', 'away_team': 'NYK', 'home_score': 115, 'away_score': 108},
])

features = fe.build_game_features(game, historical)
print(f'Features: {len(features)}')
print(f'predicted_margin: {features.get(\"predicted_margin\")}')
print(f'predicted_margin_1h: {features.get(\"predicted_margin_1h\")}')
print(f'predicted_total_1h: {features.get(\"predicted_total_1h\")}')
"
```

**Expected Output:**
```
Features: 80+
predicted_margin: -2.5
predicted_margin_1h: -1.2
predicted_total_1h: 107.8
```

**What to check:**
- ✅ `predicted_margin_1h` exists (was missing before)
- ✅ `predicted_total_1h` exists
- ✅ Features build without errors

---

### Step 5: Verify Backtest Runs

**Test:**
```bash
docker compose -f docker-compose.backtest.yml up backtest-full
```

**Expected Output:**
```
============================================================
BACKTEST: Full Game Spreads
============================================================
  Processing game 100/1200...
  Processing game 200/1200...
  ...
[OK] Completed 400+ predictions

Full Game Spreads Summary:
  Bets: 422
  Accuracy: 60.6%
  ROI: +15.7%
```

**What to check:**
- ✅ Backtest completes without errors
- ✅ Results file created: `data/processed/all_markets_backtest_results.csv`
- ✅ Report generated: `ALL_MARKETS_BACKTEST_RESULTS.md`
- ✅ Check results: `cat data/results/backtest_report_*.md`

---

### Step 6: Verify Predictions Work

**Test (if running production API):**
```bash
# Start stack
docker compose up -d

# Get predictions
curl http://localhost:8090/slate/today
```

**Expected Output:**
```json
{
  "games": [
    {
      "home_team": "LAL",
      "away_team": "BOS",
      "predictions": {
        "full_game": {
          "spread": {
            "confidence": 0.72,
            "edge": 3.5,
            "bet_side": "home"
          },
          "total": {
            "confidence": 0.68,
            "edge": 4.2,
            "bet_side": "over"
          }
        }
      }
    }
  ]
}
```

**What to check:**
- ✅ API responds
- ✅ Predictions include confidence and edge
- ✅ No errors in predictions

---

## 🔍 Quick Verification Script

Create this script to verify everything:

```bash
#!/bin/bash
# verify_stack.sh

echo "=== VERIFYING STACK ==="

# 1. Check container can start
echo "1. Testing container entry point..."
docker compose -f docker-compose.backtest.yml run --rm backtest-shell \
  python -c "from src.modeling.models import SpreadsModel; print('✓ Models import OK')"

# 2. Check data pipeline
echo "2. Testing data pipeline..."
docker compose -f docker-compose.backtest.yml run --rm backtest-data 2>&1 | grep -q "Training data saved" && \
  echo "✓ Data pipeline OK" || echo "✗ Data pipeline FAILED"

# 3. Check training data exists
echo "3. Checking training data file..."
docker compose -f docker-compose.backtest.yml run --rm backtest-shell \
  test -f /app/data/processed/training_data.csv && \
  echo "✓ Training data exists" || echo "✗ Training data missing"

# 4. Check feature engineering
echo "4. Testing feature engineering..."
docker compose -f docker-compose.backtest.yml run --rm backtest-shell \
  python -c "
from src.modeling.features import FeatureEngineer
fe = FeatureEngineer()
print('✓ FeatureEngineer OK')
features = fe.build_game_features(
    pd.Series({'home_team': 'LAL', 'away_team': 'BOS', 'date': pd.Timestamp('2025-12-18')}),
    pd.DataFrame()
)
assert 'predicted_margin_1h' in features, 'Missing 1H margin!'
print('✓ 1H features OK')
" && echo "✓ Feature engineering OK" || echo "✗ Feature engineering FAILED"

echo "=== VERIFICATION COMPLETE ==="
```

---

## 📊 Expected Results

### Training Data Stats
- **Games:** 1200+ (for 2 seasons)
- **With spread lines:** 70%+ coverage
- **With total lines:** 70%+ coverage
- **With 1H lines:** 40%+ coverage (when available)
- **With betting splits:** 20-30% coverage (when available)

### Backtest Performance (Target)
- **FG Spread:** 58-62% accuracy
- **FG Total:** 57-61% accuracy
- **1H Spread:** 55-60% accuracy
- **1H Total:** 56-60% accuracy

---

## 🚨 Common Issues & Fixes

### Issue: "Missing required API keys"
**Fix:** Ensure `.env` file has:
```
API_BASKETBALL_KEY=your_key
THE_ODDS_API_KEY=your_key
```

### Issue: "Training data not found"
**Fix:** Run data pipeline first:
```bash
docker compose -f docker-compose.backtest.yml up backtest-data
```

### Issue: "predicted_margin_1h is REQUIRED"
**Fix:** Should be fixed now - check that `features.py` includes the calculation

### Issue: "No games fetched"
**Fix:** Check API keys are valid and have quota remaining

---

## ✅ Final Verification Checklist

- [ ] Container starts without errors
- [ ] API keys validated
- [ ] Python imports work
- [ ] Games fetched from API-Basketball (check logs)
- [ ] ALL endpoints called (teams, games, statistics, game_stats_teams)
- [ ] Betting lines fetched (check logs for event-specific calls)
- [ ] 1H/Q1 markets included (check logs)
- [ ] Training data file created
- [ ] Feature engineering includes predicted_margin_1h
- [ ] Backtest runs successfully
- [ ] Results file generated
- [ ] No prediction errors

**If all checkboxes pass, the stack is working correctly!** ✅
