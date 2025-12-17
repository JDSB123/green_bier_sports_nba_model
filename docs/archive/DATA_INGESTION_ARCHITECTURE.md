# NBA Prediction System - Data Ingestion Architecture

**Version:** 4.0.0
**Last Updated:** 2025-12-06

## Overview

This document provides a comprehensive breakdown of the data ingestion modules, organization, and data flow through the NBA prediction system.

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA SOURCES                         │
├─────────────────────────────────────────────────────────────────────┤
│  The Odds API  │  API-Basketball  │  ESPN  │  Betting Splits         │
│   (Primary)    │   (Team Stats)   │(Injury)│ (Public Betting %)      │
└────────┬───────┴────────┬──────────┴───┬────┴─────────────┬──────────┘
         │                │              │                  │
         ▼                ▼              ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INGESTION LAYER (src/ingestion/)                │
├─────────────────────────────────────────────────────────────────────┤
│ • the_odds.py          - The Odds API client                         │
│ • api_basketball.py    - API-Basketball client                       │
│ • injuries.py          - ESPN injury reports                         │
│ • betting_splits.py    - Public betting percentages                  │
└────────┬────────────────────────────────────────────────────────┬────┘
         │                                                        │
         ▼                                                        ▼
┌─────────────────────────────┐        ┌─────────────────────────────┐
│  COLLECTION SCRIPTS          │        │   RAW DATA STORAGE          │
│  (scripts/)                  │───────▶│   (data/raw/)               │
├─────────────────────────────┤        ├─────────────────────────────┤
│ • collect_the_odds.py        │        │ • the_odds/                 │
│ • collect_api_basketball.py  │        │ • api_basketball/           │
│ • collect_betting_splits.py  │        │ • injuries/                 │
│ • fetch_injuries.py          │        │ • betting_splits/           │
│ • ingest_all.py (orchestrate)│        │ • fivethirtyeight/          │
└─────────────────────────────┘        └────────┬────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER (src/processing/)                 │
├─────────────────────────────────────────────────────────────────────┤
│ • Team standardization via src/utils/team_names.py                  │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING (src/modeling/)                  │
├─────────────────────────────────────────────────────────────────────┤
│ • features.py          - FeatureEngineer class (40-50 features)      │
│ • feature_config.py    - Feature configuration & definitions         │
│ • dataset.py           - Dataset building from features              │
│                                                                       │
│ HELPER SCRIPTS:                                                      │
│ • scripts/build_rich_features.py - RichFeatureBuilder (live data)    │
└────────┬────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING & SERVING                          │
├─────────────────────────────────────────────────────────────────────┤
│ • models.py            - Model definitions (Spreads, Totals, ML)     │
│ • model_tracker.py     - Version tracking & management               │
│ • io.py                - Model serialization                         │
│                                                                       │
│ SCRIPTS:                                                             │
│ • scripts/train_models.py        - Train base models                 │
│ • scripts/train_ensemble_models.py - Train ensembles                 │
│ • scripts/predict.py             - Generate predictions              │
│ • scripts/backtest.py            - Walk-forward backtesting          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔌 Ingestion Modules Detail

### 1. **The Odds API** (`src/ingestion/the_odds.py`)

**Purpose:** Primary source for betting lines, odds, and bookmaker consensus

**Endpoints:**
- `/sports/basketball_nba/events` - List of upcoming games
- `/sports/basketball_nba/odds` - Sport-level odds with bookmakers
- `/sports/basketball_nba/events/{eventId}/odds` - Per-game odds
- `/sports/basketball_nba/events/{eventId}/markets` - Per-game markets

**Data Collected:**
- Spreads (point spreads for each team)
- Totals (over/under lines)
- Moneylines (win odds)
- Bookmaker consensus (10+ sportsbooks)
- Line movement (opening vs. current)

**Key Functions:**
```python
async def fetch_odds() -> List[Dict]
async def fetch_event_odds(event_id: str) -> Dict
async def fetch_event_markets(event_id: str) -> Dict
```

**Collection Script:** `scripts/run_the_odds_tomorrow.py`
- Fetches tomorrow's games (CST timezone aware)
- Saves timestamped JSON to `data/raw/the_odds/{date}/`
- Tracks API rate limits

---

### 2. **API-Basketball** (`src/ingestion/api_basketball.py`)

**Purpose:** Comprehensive NBA statistics (teams, players, standings, H2H)

**Endpoints:**
- `/teams` - Team search and info
- `/statistics` - Season team statistics
- `/games/h2h` - Head-to-head history
- `/standings` - League standings
- `/games` - Game results and schedules

**Data Collected:**
- Team season averages (PPG, PAPG, FG%, etc.)
- Win/loss records, standings position
- Head-to-head historical matchups
- Recent game results
- Player statistics (if needed)

**Key Functions:**
```python
async def fetch_teams(search: str, league: int, season: str) -> Dict
async def fetch_statistics(league: int, season: str, team: int) -> Dict
async def fetch_h2h(h2h: str) -> Dict
async def fetch_standings(league: int, season: str) -> Dict
```

**Collection Script:** `scripts/collect_api_basketball.py`

**Caching:** Uses `data/processed/cache/` for API response caching

---

### 3. **ESPN Injuries** (`src/ingestion/injuries.py`)

**Purpose:** Free, official injury reports (no API key required)

**Endpoint:**
- `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`

**Data Collected:**
- Player injury status (Out, Doubtful, Questionable, Probable)
- Injury type/description
- Expected return date
- Team associations

**Key Functions:**
```python
async def fetch_injuries() -> List[Dict]
def parse_injury_report(data: Dict) -> pd.DataFrame
```

**Collection Script:** `scripts/fetch_injuries.py`

**Processing:**
- Maps to standardized team names via `src/utils/team_names.py`
- Estimates injury impact based on player usage rate and PPG

---

### 4. **Betting Splits** (`src/ingestion/betting_splits.py`) ✨ NEW

**Purpose:** Public betting percentages and sharp money indicators

**Sources (in order of preference):**
1. **SportsBookReview Online (SBRO)** - Free, consensus data
2. **Covers.com** - Free, requires scraping
3. **Action Network** - Paid, most reliable (not implemented)
4. **Mock Data** - Realistic simulation for development

**Data Collected:**
- Spread betting splits (% tickets on each side)
- Money percentages (% of money on each side)
- Total betting splits (over/under %)
- Line movement (opening → current)
- RLM (Reverse Line Movement) indicators
- Sharp side detection (ticket vs. money divergence)

**Key Functions:**
```python
async def fetch_public_betting_splits(games: List, source: str) -> Dict[str, GameSplits]
async def fetch_splits_sbro() -> List[GameSplits]
async def scrape_splits_covers() -> List[GameSplits]
def detect_reverse_line_movement(splits: GameSplits) -> GameSplits
def splits_to_features(splits: GameSplits) -> Dict[str, float]
def create_mock_splits(...) -> GameSplits
```

**Data Structure:**
```python
@dataclass
class GameSplits:
    event_id: str
    home_team: str
    away_team: str

    # Spread splits
    spread_home_ticket_pct: float  # % of tickets on home
    spread_home_money_pct: float   # % of money on home

    # Total splits
    over_ticket_pct: float
    over_money_pct: float

    # Line movement
    spread_open: float
    spread_current: float

    # Derived signals
    is_rlm_spread: bool           # RLM detected
    sharp_side_spread: str        # "home" or "away"
```

**Collection Script:** `scripts/collect_betting_splits.py`
```bash
python scripts/collect_betting_splits.py --source auto --save
```

**Features Extracted:** 20 features including:
- Public betting percentages
- Money percentages
- Ticket/money divergence
- RLM indicators
- Sharp side signals
- Line movement magnitude

---

## 📂 Data Storage Organization

### Raw Data (`data/raw/`)

**Structure:**
```
data/raw/
├── the_odds/
│   └── {YYYY-MM-DD}/
│       ├── events_{timestamp}.json
│       ├── sport_odds_{timestamp}.json
│       ├── event_{eventId}_odds_{timestamp}.json
│       └── event_{eventId}_markets_{timestamp}.json
├── api_basketball/
│   └── {season}/
│       ├── teams_{timestamp}.json
│       ├── statistics_{team_id}_{timestamp}.json
│       └── h2h_{team1}-{team2}_{timestamp}.json
├── injuries/
│   └── injuries_{YYYY-MM-DD}_{timestamp}.json
├── betting_splits/
│   └── splits_{YYYY-MM-DD}_{source}_{timestamp}.json
└── fivethirtyeight/
    └── nbaallelo.csv (historical training data)
```

**Characteristics:**
- ✅ Timestamped (immutable, reproducible)
- ✅ Never modified (append-only)
- ✅ Version controlled via .gitignore (excluded from repo)
- ✅ Organized by date and source

---

### Processed Data (`data/processed/`)

**Structure:**
```
data/processed/
├── cache/
│   ├── stats_{league}_{season}_{team_id}.joblib
│   ├── h2h_{team1}-{team2}.joblib
│   └── standings_{league}_{season}.joblib
├── models/
│   ├── spreads_model.joblib
│   ├── totals_model.joblib
│   ├── moneyline_model.joblib
│   ├── ensemble_spreads.joblib
│   ├── manifest.json (model version tracking)
│   └── production.json (active model pointers)
├── predictions.csv
├── betting_splits.json
├── training_data.csv
└── backtest_results.csv
```

**Characteristics:**
- ✅ Cached API responses (performance optimization)
- ✅ Trained model artifacts
- ✅ Prediction outputs
- ✅ Intermediate datasets

---

## 🔄 Data Flow Pipeline

### 1. **Collection Phase**

**Orchestration Script:** `scripts/ingest_all.py`

```python
# Fetch all data sources in parallel
await asyncio.gather(
    collect_the_odds(),
    collect_api_basketball(),
    fetch_injuries(),
    collect_betting_splits(),
)
```

**Output:** Raw JSON files in `data/raw/`

---

### 2. **Processing Phase**

**Team Name Standardization:** `src/utils/team_names.py`

```python
from src.utils.team_names import reconcile_team_name

# Map various API naming conventions to standard names
standard_name = reconcile_team_name("LA Lakers")
# Returns: "Los Angeles Lakers"
```

**Uses:** RapidFuzz for fuzzy string matching

**Supported Variations:**
- "LA Lakers" → "Los Angeles Lakers"
- "Sixers" → "Philadelphia 76ers"
- "Heat" → "Miami Heat"

---

### 3. **Feature Engineering Phase**

**Main Class:** `FeatureEngineer` (`src/modeling/features.py`)

**Input Sources:**
- Historical game data (DataFrame)
- Odds data (DataFrame or API response)
- Injury reports (DataFrame)
- Betting splits (GameSplits objects)

**Feature Categories (40-50 features):**

1. **Team Rolling Statistics (10 games lookback)**
   - Points per game (PPG)
   - Points allowed per game (PAPG)
   - Average margin
   - Win percentage
   - Home/away splits
   - ATS (Against The Spread) performance

2. **Rest & Scheduling**
   - Days of rest
   - Back-to-back detection
   - Rest advantage differential

3. **Head-to-Head History**
   - Win rate in recent matchups
   - Average margin in H2H games

4. **Market-Derived Features**
   - Spread line (consensus)
   - Total line (consensus)
   - Line vs. predicted margin divergence
   - Bookmaker disagreement (std dev)

5. **Line Movement & RLM**
   - Opening vs. current line
   - Line movement magnitude
   - RLM binary indicators
   - Sharp side signals

6. **Betting Splits** ✨ NEW (14 features)
   - Public betting percentages
   - Money percentages
   - Ticket/money divergence
   - RLM indicators
   - Sharp side indicators

7. **Injury Impact**
   - Players out (count)
   - Star player out (binary)
   - Estimated point impact
   - Net injury differential

8. **Derived Predictive Features**
   - Predicted margin (home perspective)
   - Predicted total
   - ELO ratings (proxy)
   - Strength differentials

**Usage:**
```python
from src.modeling.features import FeatureEngineer, BettingSplits

engineer = FeatureEngineer()
features = engineer.build_game_features(
    game_row=game,
    historical_games=historical_df,
    odds_df=odds_df,
    injuries_df=injuries_df,
    betting_splits=splits,  # NEW
)
```

---

### 4. **Live Feature Building**

**Class:** `RichFeatureBuilder` (`scripts/build_rich_features.py`)

**Purpose:** Build features from live API data (no historical DataFrame)

**Features:**
- Fetches data directly from APIs
- Caches responses for performance
- Integrates betting splits
- Returns feature dict ready for model

**Usage:**
```python
from scripts.build_rich_features import RichFeatureBuilder

builder = RichFeatureBuilder(league_id=12, season="2025-2026")
features = await builder.build_game_features(
    home_team="Los Angeles Lakers",
    away_team="Boston Celtics",
    betting_splits=splits,  # Optional
)
```

**Integration Point:** Used by `scripts/predict.py` for live predictions

---

### 5. **Model Training Phase**

**Scripts:**
- `scripts/train_models.py` - Train base models (Logistic, GradientBoosting)
- `scripts/train_ensemble_models.py` - Train ensemble combinations

**Input:** `data/processed/training_data.csv` (generated from FiveThirtyEight data)

**Models Trained:**
- SpreadsModel (binary: home covers spread?)
- TotalsModel (binary: over total?)
- MoneylineModel (binary: home wins?)

**Output:** `data/processed/models/*.joblib`

**Tracking:** `manifest.json` maintains version history

---

### 6. **Prediction Phase**

**Script:** `scripts/predict.py`

**Flow:**
```
1. Fetch upcoming games (The Odds API)
2. Fetch betting splits (auto-source selection)
3. Build rich features (RichFeatureBuilder + splits)
4. Load trained models
5. Generate predictions
6. Calculate edge vs. Vegas lines
7. Determine bet recommendations
8. Save to predictions.csv
```

**Command:**
```bash
# With betting splits (default)
python scripts/predict.py --date tomorrow

# Without betting splits
python scripts/predict.py --date tomorrow --no-betting-splits
```

**Output:** `data/processed/predictions.csv`

**Columns:**
- date, home_team, away_team
- spread_line, predicted_margin, edge
- bet_side, confidence
- predicted_total
- home_ppg, away_ppg, home_elo, away_elo

---

## 🧪 Testing & Validation

### Smoke Tests

**Script:** `scripts/smoke_ingestion_endpoints.py`

**Purpose:** Verify all API endpoints are accessible

**Tests:**
- The Odds API connectivity
- API-Basketball authentication
- ESPN injuries availability

---

### Integration Tests

**Script:** `scripts/test_betting_splits_integration.py`

**Tests:**
1. Mock splits generation
2. RLM detection accuracy
3. Feature conversion (20 features)
4. Full integration with FeatureEngineer

**Command:**
```bash
python scripts/test_betting_splits_integration.py
```

**Expected:** ✓ ALL TESTS PASSED

---

### Validation Scripts

**Script:** `scripts/validate_leakage.py`

**Purpose:** Ensure no data leakage in feature engineering

**Checks:**
- Features only use past data
- No lookahead bias
- Temporal split integrity

---

## 🔑 Environment Configuration

### Required Environment Variables

**File:** `.env` (create locally, not in repo)

```bash
# Required for betting lines
THE_ODDS_API_KEY=your_key_here

# Required for team statistics
API_BASKETBALL_KEY=your_key_here

# Optional (for future integrations)
# ACTION_NETWORK_API_KEY=your_key_here
# BETSAPI_KEY=your_key_here
```

### Configuration Module

**File:** `src/config.py`

**Settings:**
```python
from src.config import settings

settings.the_odds_api_key        # The Odds API key
settings.api_basketball_key      # API-Basketball key
settings.current_season          # "2025-2026"
settings.data_raw_dir            # "data/raw"
settings.data_processed_dir      # "data/processed"
```

---

## 📊 Module Dependency Graph

```
┌─────────────────┐
│  Collection     │
│  Scripts        │
│  (scripts/)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Ingestion      │────▶│  Processing  │
│  (src/ingestion)│     │ (src/process)│
└────────┬────────┘     └──────┬───────┘
         │                     │
         │  ┌──────────────────┘
         │  │
         ▼  ▼
┌─────────────────┐     ┌──────────────┐
│  Utils          │────▶│  Feature Eng │
│ (src/utils)     │     │ (src/modeling│
└─────────────────┘     │  /features)  │
                        └──────┬───────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Models      │
                        │ (src/modeling│
                        │  /models)    │
                        └──────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### 2. Set Environment Variables

```bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
THE_ODDS_API_KEY=your_key
API_BASKETBALL_KEY=your_key
```

### 3. Collect Data

```bash
# Collect all data sources
python scripts/ingest_all.py

# Or individually
python scripts/run_the_odds_tomorrow.py
python scripts/collect_betting_splits.py --save
python scripts/fetch_injuries.py
```

### 4. Train Models

```bash
# Generate training data from FiveThirtyEight
python scripts/generate_training_data.py

# Train base models
python scripts/train_models.py

# Train ensembles
python scripts/train_ensemble_models.py
```

### 5. Generate Predictions

```bash
# Predict tomorrow's games (with betting splits)
python scripts/predict.py --date tomorrow

# Predict specific date
python scripts/predict.py --date 2025-12-10
```

### 6. Backtest

```bash
# Run walk-forward backtest
python scripts/backtest.py
```

---

## 📈 Data Freshness & Updates

### Recommended Collection Schedule

**Daily (automated via cron/scheduler):**
- The Odds API: 6 AM, 12 PM, 6 PM (line movement tracking)
- Betting Splits: 12 PM, 6 PM (sharp money detection)
- Injuries: 8 AM, 4 PM (morning shootaround, pregame updates)

**As Needed:**
- API-Basketball: Weekly (season stats update slowly)
- Training Data: Monthly (retrain models with recent games)

### Data Staleness Handling

**Caching Strategy:**
- API-Basketball: 24-hour cache (stats don't change quickly)
- The Odds API: No cache (lines move constantly)
- Injuries: 6-hour cache (updated twice daily)
- Betting Splits: 2-hour cache (move throughout day)

---

## 🔧 Troubleshooting

### Common Issues

**1. API Rate Limits**
- The Odds API: 500 requests/month (monitor via headers)
- API-Basketball: 100 requests/day (cache aggressively)
- Solution: Use caching, batch requests

**2. Team Name Mismatches**
- Symptom: Features missing for some games
- Solution: Check `src/utils/team_names.py`, add mapping
- Tool: RapidFuzz fuzzy matching (threshold=80)

**3. Missing Betting Splits**
- Symptom: RLM features = 0
- Solution: Check source availability, fallback to mock
- Current: Auto-fallback SBRO → Covers → Mock

**4. Data Leakage**
- Symptom: Unrealistic backtest accuracy (>70%)
- Solution: Run `scripts/validate_leakage.py`
- Check: Temporal splits, feature availability

---

## 📚 Additional Resources

### Documentation Files
- `DATA_FLOW_GUIDE.md` - Detailed data flow documentation
- `IMPLEMENTATION_SUMMARY.md` - Feature implementation details
- `CLEANUP_AND_INTEGRATION_SUMMARY.md` - Recent changes log
- `QUICK_REFERENCE.md` - Command quick reference

### Code References
- Team Name Reconciliation: `src/utils/team_names.py`
- Feature Definitions: `src/modeling/feature_config.py`
- Model I/O: `src/modeling/io.py`
- Model Versioning: `src/modeling/model_tracker.py`

---

**End of Data Ingestion Architecture Document**
