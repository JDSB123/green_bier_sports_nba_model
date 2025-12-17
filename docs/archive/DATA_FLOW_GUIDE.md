# NBA V4.0 - Complete Data Flow Guide

## 📊 What Data Goes Where - Complete Breakdown

---

## 🔄 Data Flow Overview

```
┌─────────────────┐
│   DATA SOURCES  │
└────────┬────────┘
         │
         ├──► The Odds API ──────► RAW odds data
         ├──► ESPN API ───────────► RAW injury data
         ├──► API-Basketball ────► RAW player stats
         └──► FiveThirtyEight ───► Historical training data
                 │
                 ▼
┌────────────────────────────┐
│   PROCESSING SCRIPTS       │
└──────────┬─────────────────┘
           │
           ├──► Process odds data ──► PROCESSED splits, lines
           ├──► Enrich injuries ────► PROCESSED injury impact
           └──► Generate features ──► PROCESSED training data
                 │
                 ▼
┌────────────────────────────┐
│   MODELING PIPELINE        │
└──────────┬─────────────────┘
           │
           ├──► Train base models ──► MODELS (spreads, totals, ML)
           ├──► Train ensembles ────► MODELS (ensemble)
           └──► Track performance ──► MODELS (manifest.json)
                 │
                 ▼
┌────────────────────────────┐
│   PREDICTIONS OUTPUT       │
└────────────────────────────┘
           │
           └──► predictions.csv
```

---

## 📂 Directory Structure & Data Locations

### **data/raw/** - Raw API responses (never modified)

```
data/raw/
├── the_odds/
│   └── YYYY-MM-DD/              # Date-specific odds data
│       ├── events_*.json         # All NBA events for the day
│       ├── sport_odds_*.json     # Sport-level odds (all games)
│       ├── participants_*.json   # NBA team list
│       ├── scores_*.json         # Recent game scores
│       ├── upcoming_odds_*.json  # Cross-sport upcoming events
│       ├── event_{id}_odds_*.json      # Per-game odds
│       ├── event_{id}_markets_*.json   # Per-game markets
│       └── summary_*.json        # Summary metadata
│
├── api_basketball/          # API-Basketball responses (if used)
├── betsapi/                 # BetsAPI responses (if used)
└── optional/                # Optional sources (if enabled)
```

### **data/processed/** - Processed, model-ready data

```
data/processed/
├── injuries.csv              # Current injury reports
├── betting_splits.csv        # Line movement and RLM signals
├── first_half_lines.csv      # First-half spreads/totals
├── team_totals_lines.csv     # Team-specific totals
├── training_data.csv         # Historical games for training
├── historical_games.csv      # Game results archive
├── predictions.csv           # Current predictions
│
└── models/                   # Trained models
    ├── spreads_model.joblib
    ├── totals_model.joblib
    ├── moneyline_model.joblib
    ├── spreads_ensemble.joblib
    ├── totals_ensemble.joblib
    ├── fh_spreads_model.joblib
    ├── fh_totals_model.joblib
    ├── manifest.json         # Model tracking
    └── production.json       # Production model pointers
```

---

## 🎯 Segment-by-Segment Data Breakdown

### **SEGMENT 1: The Odds API Raw Data**

**Script**: `scripts/run_the_odds_tomorrow.py`

**What it fetches**:

#### A. Sport-Level Endpoints (All Games)
1. **events_*.json**
   - All NBA games (past, present, future)
   - Fields: event_id, home_team, away_team, commence_time, sport_key
   - Used for: Finding tomorrow's games

2. **sport_odds_*.json**
   - All NBA games WITH odds from bookmakers
   - Fields: event data + bookmakers array
   - Each bookmaker has: markets (spreads, totals, h2h)
   - Used for: Getting consensus lines, comparing bookmakers

3. **participants_*.json**
   - List of all 30 NBA teams
   - Fields: id, name, team
   - Used for: Team name validation

4. **scores_*.json**
   - Recent game results
   - Fields: event data + scores + completed flag
   - Used for: Validation, recent form

5. **upcoming_odds_*.json**
   - Cross-sport upcoming events (NBA + other sports)
   - Fields: Similar to sport_odds but multi-sport
   - Used for: Alternative data source

#### B. Event-Level Endpoints (Per Game)
For each game tomorrow:

6. **event_{id}_odds_*.json**
   - Single game odds from all bookmakers
   - Fields: event info + bookmakers array
   - Each market has outcomes with prices/points
   - Used for: Detailed per-game odds analysis

7. **event_{id}_markets_*.json**
   - All available markets for a single game
   - Fields: event info + bookmakers + ALL markets
   - Includes: player props, quarters, halves, team totals, etc.
   - Used for: Extracting first-half lines, team totals

#### C. Summary Metadata
8. **summary_*.json**
   - Aggregated metadata for the fetch
   - Fields: target_date, events_count, files_list, rate_limits
   - Used for: Tracking, debugging, API quota monitoring

**Example Data Sizes** (for 5 games tomorrow):
- events: ~13 total games (includes other dates)
- sport_odds: ~13 games with odds
- participants: 30 teams
- scores: ~19 recent games
- upcoming: ~46 cross-sport events
- Per-game odds: 5 files
- Per-game markets: 5 files
- **Total: ~17 JSON files**

---

### **SEGMENT 2: Injury Data**

**Script**: `scripts/fetch_injuries.py`

**What it fetches**:

#### A. ESPN Injuries (Free, No API Key)
- Endpoint: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`
- Fields: player_name, team, status, injury_type
- Updates: Daily
- Reliability: High (official source)

#### B. API-Basketball Injuries (Requires Key)
- Endpoint: `/injuries?league=12&season=2024-2025`
- Fields: player_id, player_name, team, status, injury_type, date
- Updates: Real-time
- Additional: More detailed injury info

**Output**: `data/processed/injuries.csv`

**Columns**:
```
player_id, player_name, team, team_id,
status (out/doubtful/questionable/probable),
injury_type, injury_location,
report_date, expected_return,
ppg, minutes_per_game, usage_rate,
source (espn/api_basketball)
```

**Example**:
```
player_name          team      status  ppg   injury_type
LeBron James        Lakers    out     25.3  Ankle
Stephen Curry       Warriors  doubt   28.1  Shoulder
```

**Used For**:
- Injury impact features in models
- Adjusting predictions for missing players
- Identifying high-impact injuries

---

### **SEGMENT 3: Processed Odds Data**

**Script**: `scripts/process_odds_data.py`

**Input**: `data/raw/the_odds/YYYY-MM-DD/*.json`

**Outputs**:

#### A. betting_splits.csv
Extracted from comparing bookmakers in sport_odds_*.json

**Columns**:
```
event_id, home_team, away_team, game_time,
spread_line (consensus), spread_open, spread_current, spread_movement,
spread_line_std (disagreement between books),
total_line (consensus), total_open, total_current, total_movement,
total_line_std,
is_rlm_spread, is_rlm_total,
sharp_spread_side, sharp_total_side,
bookmaker_count, source
```

**Example**:
```
event_id     home_team  away_team  spread_line  spread_open  spread_current  spread_movement
abc123       Lakers     Celtics    -5.5         -6.0         -5.5            +0.5
```

**Logic**:
- Finds earliest bookmaker update (opening line)
- Finds latest bookmaker update (current line)
- Movement = current - open
- RLM = public on one side but line moves opposite (requires public % data)

**Used For**:
- RLM features in models
- Line movement features
- Sharp money indicators

#### B. first_half_lines.csv
Extracted from event_{id}_markets_*.json

**Columns**:
```
event_id, home_team, away_team, commence_time,
fh_spread_line, fh_spread_line_std, fh_spread_bookmaker_count,
fh_total_line, fh_total_line_std, fh_total_bookmaker_count
```

**Example**:
```
event_id  home_team  fh_spread_line  fh_total_line
abc123    Lakers     -3.0            110.5
```

**Used For**:
- First-half model predictions
- 1H spreads betting
- 1H totals betting

**Status**: Framework complete, data availability depends on The Odds API

#### C. team_totals_lines.csv
Extracted from event_{id}_markets_*.json

**Columns**:
```
event_id, home_team, away_team, commence_time,
home_team_total_line, home_team_total_bookmaker_count,
away_team_total_line, away_team_total_bookmaker_count
```

**Example**:
```
event_id  home_team  home_team_total_line  away_team_total_line
abc123    Lakers     112.5                 108.5
```

**Used For**:
- Team totals model predictions
- Individual team scoring bets

---

### **SEGMENT 4: Training Data**

**Script**: `scripts/generate_training_data.py`

**Input Sources**:
1. FiveThirtyEight ELO dataset (primary)
   - URL: `https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv`
   - Contains: Historical game results, ELO ratings
   - Coverage: Last 5 seasons

2. Synthetic data (fallback if FiveThirtyEight unavailable)
   - Generates ~2000 simulated games
   - Uses realistic team strengths

**Output**: `data/processed/training_data.csv`

**Columns** (40+ features):
```
# Identifiers
source, game_id, date, season, home_team, away_team

# Actual Results (targets)
home_score, away_score, total_score, home_margin

# Betting Lines (simulated based on ELO)
spread_line, total_line

# Targets for Classification
spread_covered (1/0), went_over (1/0),
spread_push (1/0), total_push (1/0)

# ELO Features
home_elo, away_elo, elo_diff, elo_prob_home

# Rolling Team Stats (computed)
home_ppg, home_papg, home_avg_margin,
away_ppg, away_papg, away_avg_margin,
home_total_ppg, away_total_ppg

# Derived Features
predicted_margin, predicted_total,
win_pct_diff, ppg_diff
```

**Example Row**:
```
date        home_team  away_team  home_score  away_score  spread_line  spread_covered  home_ppg  away_ppg
2024-11-15  Lakers     Celtics    112         108         -3.5         1               110.5     108.2
```

**Used For**:
- Training all models (spreads, totals, moneyline, ensembles)
- Backtesting
- Feature importance analysis

---

### **SEGMENT 5: Models**

**Scripts**:
- `scripts/train_models.py` - Base models
- `scripts/train_ensemble_models.py` - Ensemble models

**Input**: `data/processed/training_data.csv`

**Outputs**: `data/processed/models/*.joblib`

#### Model Files:

1. **spreads_model.joblib**
   - Type: Logistic Regression or Gradient Boosting
   - Predicts: Will home team cover the spread? (1/0)
   - Features: ~20-30 (filtered from available)
   - Output: Binary classification + probability

2. **totals_model.joblib**
   - Type: Logistic Regression or Gradient Boosting
   - Predicts: Will game go over total? (1/0)
   - Features: ~15-25
   - Output: Binary classification + probability

3. **moneyline_model.joblib**
   - Type: Logistic Regression
   - Predicts: Will home team win? (1/0)
   - Features: ~10-15
   - Output: Win probability

4. **fh_spreads_model.joblib**
   - Type: Same as spreads
   - Predicts: Will home team cover first-half spread?
   - Features: Same as full-game spreads
   - Note: Requires first-half score data in training

5. **fh_totals_model.joblib**
   - Type: Same as totals
   - Predicts: Will first-half go over?
   - Features: Same as full-game totals

6. **spreads_ensemble.joblib**
   - Type: Ensemble (Logistic + GB)
   - Contains: Both models + weights
   - Typically: 2-5% better accuracy than single model

7. **totals_ensemble.joblib**
   - Type: Ensemble (Logistic + GB)
   - Contains: Both models + weights

#### Model Tracking:

8. **manifest.json**
   ```json
   {
     "versions": [
       {
         "version": "1.1.0",
         "model_type": "spreads",
         "algorithm": "ensemble",
         "trained_at": "2024-12-03T...",
         "train_samples": 1500,
         "test_samples": 375,
         "features_count": 25,
         "metrics": {
           "accuracy": 0.589,
           "log_loss": 0.654
         },
         "file_path": "spreads_ensemble.joblib"
       }
     ],
     "active_models": {
       "spreads": "1.1.0",
       "totals": "1.1.0"
     },
     "performance": []
   }
   ```

9. **production.json**
   ```json
   {
     "version": "1.1.0",
     "model_type": "spreads",
     "promoted_at": "2024-12-03T...",
     "file_path": "spreads_ensemble.joblib"
   }
   ```

---

### **SEGMENT 6: Predictions**

**Script**: `scripts/predict.py`

**Inputs**:
1. `data/processed/models/spreads_model.joblib` (or ensemble)
2. `data/processed/models/totals_model.joblib` (or ensemble)
3. Live odds data from The Odds API
4. `data/processed/injuries.csv`
5. `data/processed/betting_splits.csv`

**Output**: `data/processed/predictions.csv`

**Columns**:
```
date (CST format),
home_team, away_team,
predicted_spread, confidence (probability),
predicted_total,
home_ppg, away_ppg,
home_elo, away_elo
```

**Example**:
```
date                      home_team  away_team  predicted_spread  confidence  predicted_total
2024-12-04 18:10:00 CST   Wizards    Celtics    +8.5              0.623       225.3
```

**Used For**:
- Daily betting decisions
- Value bet identification
- Performance tracking

---

## 🔄 Complete Data Flow Example

### Scenario: Predicting tomorrow's Lakers vs Celtics game

#### Step 1: Fetch Odds Data
```bash
python scripts/run_the_odds_tomorrow.py
```
**Creates**:
- `data/raw/the_odds/2024-12-04/events_*.json` → Contains Lakers vs Celtics event
- `data/raw/the_odds/2024-12-04/event_abc123_odds_*.json` → Lakers vs Celtics odds
- `data/raw/the_odds/2024-12-04/event_abc123_markets_*.json` → All markets

**Data Extracted**:
- Event ID: abc123
- Home: Lakers
- Away: Celtics
- Time: 2024-12-04 19:00:00 UTC
- Spread consensus: Lakers -5.5
- Total consensus: 222.5
- Bookmakers: 10

#### Step 2: Fetch Injuries
```bash
python scripts/fetch_injuries.py
```
**Creates**: `data/processed/injuries.csv`

**Data Found**:
- LeBron James, Lakers, out, 25.3 PPG → High impact!
- Jayson Tatum, Celtics, questionable, 28.1 PPG

#### Step 3: Process Odds
```bash
python scripts/process_odds_data.py
```
**Creates**: `data/processed/betting_splits.csv`

**For Lakers vs Celtics**:
- Spread opened: -6.0
- Spread current: -5.5
- Movement: +0.5 (line moved toward Celtics)
- Possible RLM: Public likely on Lakers but line moved toward Celtics

#### Step 4: Generate Predictions
```bash
python scripts/predict.py
```

**Process**:
1. Load models (spreads_ensemble.joblib, totals_ensemble.joblib)
2. Fetch game data from The Odds API
3. Build features:
   - Lakers PPG: 115.2
   - Celtics PPG: 118.5
   - Lakers injury impact: -25.3 (LeBron out!)
   - Spread line: -5.5
   - Line movement: +0.5
   - ELO ratings from historical data

4. Make predictions:
   - Spread: Celtics likely to cover (Lakers -5.5 → Celtics +5.5)
   - Confidence: 64%
   - Total: Under 222.5
   - Confidence: 58%

**Creates**: `data/processed/predictions.csv`
```
date                     home_team  away_team  predicted_spread  confidence  predicted_total
2024-12-04 19:00 CST     Lakers     Celtics    +5.5 (Celtics)   0.64        219.8 (Under)
```

---

## 📊 Data Size & Performance

### Typical Data Volumes

**Daily Odds Fetch** (5 games):
- Total JSON files: ~17
- Total size: ~2-5 MB
- API calls: ~15-20
- Time: ~30 seconds

**Injury Fetch**:
- Players: ~100-150
- Size: ~50 KB
- API calls: 1-2
- Time: ~5 seconds

**Training Data**:
- Games: ~1500-2000
- Size: ~2 MB
- Features: 40+
- Time to generate: ~60 seconds

**Model Training**:
- Spreads model: ~30 seconds
- Totals model: ~30 seconds
- Ensemble: ~60 seconds
- Total: ~2 minutes

**Predictions**:
- Games: 5-15 per day
- Time: ~10 seconds
- Output size: ~5 KB

---

## 🎯 Key Takeaways

### What Data is Most Important?

1. **The Odds API sport_odds_*.json** → Most critical
   - Contains all lines, bookmakers, and odds
   - Used for everything

2. **injuries.csv** → High impact
   - Missing stars = huge line movement
   - Critical for accurate predictions

3. **betting_splits.csv** → Sharp money indicator
   - Identifies value bets
   - RLM signals

4. **training_data.csv** → Foundation
   - Models can't work without this
   - More data = better models

### What Can Be Skipped?

- upcoming_odds (redundant with sport_odds)
- scores (nice to have, not critical)
- first_half_lines (if not betting 1H)
- team_totals (if not betting team totals)

---

This guide shows EXACTLY what data goes where and how it all connects!

