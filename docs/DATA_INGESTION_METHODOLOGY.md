# Data Ingestion Methodology

**Last Updated:** 2025-12-20
**Version:** 6.0 - Production Hardened

---

## Overview

The NBA v5.0 BETA system uses a **multi-source data ingestion pipeline** that collects data from various APIs, standardizes it to a canonical ESPN format, validates it, and stores it for model training and prediction.

### Key Principles

1. **ESPN as Single Source of Truth** - All team names standardized to ESPN format
2. **No Fake Data Policy** - Invalid/unstandardized data is rejected, not passed through
3. **Validation at Ingestion** - Data quality checks happen immediately upon ingestion
4. **Multi-Source Aggregation** - Data from multiple APIs merged with validation flags

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ The Odds API │  │API-Basketball│  │ ESPN (Free)  │         │
│  │              │  │              │  │              │         │
│  │ • Live Odds  │  │ • Game Scores│  │ • Schedule   │         │
│  │ • Spreads    │  │ • Box Scores │  │ • Standings  │         │
│  │ • Totals     │  │ • Statistics │  │              │         │
│  │ • Moneyline  │  │ • H2H        │  │              │         │
│  │ • Betting    │  │ • Injuries   │  │              │         │
│  │   Lines      │  │ • Rosters    │  │              │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                 │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                   ┌─────────▼─────────┐
                   │  STANDARDIZATION  │
                   │   (standardize.py)│
                   │                   │
                   │ • Team Names → ESPN│
                   │ • Date Normalization│
                   │ • Validation Flags│
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │  DATA VALIDATION  │
                   │                   │
                   │ • _data_valid     │
                   │ • _home_team_valid│
                   │ • _away_team_valid│
                   │ • Reject invalid  │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │  RAW STORAGE      │
                   │  (data/raw/)      │
                   │                   │
                   │ • JSON files      │
                   │ • Timestamped     │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │  PROCESSING       │
                   │  (CSV conversion) │
                   │                   │
                   │ • odds_the_odds.csv│
                   │ • game_outcomes.csv│
                   │ • betting_splits.csv│
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │  TRAINING DATA    │
                   │  (build_training) │
                   │                   │
                   │ • Link odds→outcomes│
                   │ • Feature engineering│
                   │ • training_data.csv│
                   └───────────────────┘
```

---

## Data Sources

### 1. The Odds API (`src/ingestion/the_odds.py`)

**Purpose:** Live betting odds and lines

**Production Status:** ✅ HARDENED (v6.0)

**All 11 Endpoints (fully hardened):**

| Function | API Endpoint | Purpose | Cost |
|----------|--------------|---------|------|
| `fetch_sports` | `/sports` | List all available sports | FREE |
| `fetch_odds` | `/sports/{sport}/odds` | Live odds for upcoming games | 1 credit |
| `fetch_event_odds` | `/events/{id}/odds` | All markets for single event | 1 credit |
| `fetch_event_markets` | `/events/{id}/markets` | Available markets per bookmaker | 1 credit |
| `fetch_events` | `/sports/{sport}/events` | List of upcoming events | FREE |
| `fetch_scores` | `/sports/{sport}/scores` | Live and recent scores | 1-2 credits |
| `fetch_historical_odds` | `/historical/.../odds` | Historical odds snapshot | 10 credits |
| `fetch_historical_events` | `/historical/.../events` | Historical events list | 1 credit |
| `fetch_historical_event_odds` | `/historical/.../events/{id}/odds` | Historical odds for single event | 10 credits |
| `fetch_betting_splits` | `/betting-splits` | Public betting percentages | ⚠️ UNDOCUMENTED |
| `fetch_participants` | `/participants` | List teams in sport | 1 credit |

**Hardening Applied to ALL Endpoints:**
- ✅ API key validation (raises `ValueError` if missing)
- ✅ Circuit breaker pattern (prevents cascading failures)
- ✅ Mandatory team name standardization to ESPN format
- ✅ **ZERO FALLBACK** - Invalid/unstandardized data is SKIPPED, never appended
- ✅ Retry with exponential backoff (3 attempts)
- ✅ Proper logging (no print statements)

**Data Collected:**
- Full-game spread and total lines
- First-half spread and total lines
- First-quarter spread and total lines
- Player props and alternate lines (via fetch_event_odds)
- Available markets per bookmaker (via fetch_event_markets)
- Sportsbook metadata + snapshot timestamps

**Standardization:**
- Team names normalized to ESPN format
- Invalid games filtered out (returns empty list)
- Validation flags added (`_data_valid`, `_home_team_valid`, `_away_team_valid`)

**Error Handling:**
- Retry with exponential backoff (3 attempts)
- Circuit breaker opens after repeated failures
- Invalid team names logged at ERROR level
- Skipped games tracked and reported in logs

**Historical Line Capture:**
- `scripts/collect_historical_lines.py` captures daily snapshots (FG/1H markets) via the historical odds endpoint and stores them under `data/raw/the_odds/historical/`.
- `scripts/extract_betting_lines.py` converts the raw snapshots into consensus lines (`data/processed/betting_lines.csv`) using median aggregation per market.

---

### 2. API-Basketball (`src/ingestion/api_basketball.py`)

**Purpose:** Game outcomes, statistics, and team data

**Production Status:** ✅ HARDENED (v6.0)

**Hardening Applied:**
- ✅ API key validation in `__init__` (raises `ValueError` if missing)
- ✅ Circuit breaker pattern in `_fetch` method
- ✅ All `print()` statements replaced with proper `logger` calls
- ✅ Team name standardization in `fetch_games` (mandatory)
- ✅ Invalid games skipped, not appended

**Endpoint Tiers:**

#### TIER 1 - ESSENTIAL (always ingest)
- `/teams` - Team reference data (34 NBA teams)
- `/games` - Game schedule with Q1-Q4 scores (core data)
- `/statistics` - Team PPG, PAPG, W-L records (key features)
- `/games/statistics/teams` - Full box scores per game (advanced stats)

#### TIER 2 - VALUABLE (optional, richer features)
- `/standings` - Conference/Division standings & rankings
- `/games?h2h` - Head-to-head history between teams
- `/games/statistics/players` - Player-level box scores

#### TIER 3 - REFERENCE (static, ingest occasionally)
- `/players` - Team rosters (rarely changes mid-season)
- `/bookmakers` - Sportsbook names (static reference)
- `/bets` - Bet type definitions (static reference)

**Data Collected:**
- Game scores (full game and by quarter)
- Team statistics (PPG, PAPG, FG%, etc.)
- Player statistics
- Head-to-head records
- Injury reports

**Standardization:**
- Team names normalized to ESPN format
- Games without valid team names skipped
- Validation flags added

**Modes:**
- `ingest_essential()` - TIER 1 only (faster, fewer API calls)
- `ingest_all()` - All tiers (slower, more comprehensive)

---

### 3. ESPN (`src/ingestion/espn.py`)

**Purpose:** Schedule and standings (free source)

**Data Collected:**
- NBA schedule
- Game dates and matchups
- Standings (used as reference for standardization)

---

### 4. Betting Splits (`src/ingestion/betting_splits.py`)

**Purpose:** Public betting percentages and reverse line movement

**Production Status:** ✅ HARDENED (v6.0)

**Hardening Applied:**
- ✅ All `print()` statements replaced with `logger.warning()` calls
- ✅ Credential validation before Action Network auth attempt
- ✅ Team name standardization (mandatory)
- ✅ Invalid splits return `None` (skipped)
- ✅ Graceful degradation when sources unavailable

**Sources:**
- Action Network (if credentials provided)
- The Odds API splits (via `fetch_betting_splits`)
- SBRO (if available)
- Covers.com (if available)

**Data Collected:**
- Public betting percentages (% of bets on each side)
- Reverse line movement (RLM) signals
- Sharp vs. public money indicators
- Ticket vs. money divergence

**Standardization:**
- Team names normalized to ESPN format
- Invalid splits return `None` (skipped)

---

### 5. Injuries (`src/ingestion/injuries.py`)

**Purpose:** Player injury status

**Sources:**
- ESPN (free)
- API-Basketball (if key available)

**Data Collected:**
- Player injury status
- Injury type/description
- Team impact

---

## Standardization Process

### Team Name Standardization (`src/ingestion/standardize.py`)

**Canonical Format:** ESPN team names

**Function:** `normalize_team_to_espn(team_name, source) -> (normalized_name, is_valid)`

**Process:**
1. **Input Validation:**
   - Empty/whitespace → Returns `("", False)`
   - None → Returns `("", False)`

2. **Matching Strategy (in order):**
   - Exact match against `ESPN_TEAM_NAMES` set
   - Lookup in `TEAM_NAME_MAPPING` dictionary
   - Abbreviation matching (e.g., "LAL" → "Los Angeles Lakers")
   - Fuzzy matching (Levenshtein distance)

3. **Output:**
   - **Success:** `(normalized_name, True)`
   - **Failure:** `("", False)` + ERROR log

**Key Feature:** Returns `("", False)` on failure instead of original name to prevent fake data.

---

### Game Data Standardization (`standardize_game_data()`)

**Function:** `standardize_game_data(game_data, source) -> standardized_game`

**Process:**
1. Extract `home_team` and `away_team` from various input formats
2. Normalize both team names using `normalize_team_to_espn()`
3. Add validation flags:
   - `_home_team_valid` - Home team name valid
   - `_away_team_valid` - Away team name valid
   - `_data_valid` - Both teams valid (game can be used)

4. Normalize date to ISO format
5. Standardize field names to canonical format

**Output Format:**
```python
{
    "home_team": "Los Angeles Lakers",  # ESPN format or ""
    "away_team": "Boston Celtics",      # ESPN format or ""
    "date": "2025-12-17T19:00:00Z",
    "_home_team_valid": True,
    "_away_team_valid": True,
    "_data_valid": True,
    # ... other fields ...
}
```

---

## Validation & Error Handling

### No Fake Data Policy

**Rule:** Invalid data is **rejected**, not passed through with warnings.

**Implementation:**
- Invalid team names return empty string `""`, not original name
- Games with `_data_valid=False` are skipped in ingestion
- All failures logged at ERROR level (not WARNING)

**Benefits:**
- Prevents silent data corruption
- Forces explicit handling of data quality issues
- Makes problems visible in logs immediately

---

### Validation Flags

All standardized game data includes validation flags:

- `_data_valid: bool` - Can this game be used for predictions?
- `_home_team_valid: bool` - Is home team name valid?
- `_away_team_valid: bool` - Is away team name valid?

**Usage:**
```python
if standardized_game.get("_data_valid"):
    # Process game
else:
    # Skip game - log error
```

---

## Ingestion Workflow

### Step 1: Fetch Raw Data

**Script:** `scripts/ingest_all.py`

**Execution:**
```bash
# Full ingestion (all tiers)
python scripts/ingest_all.py

# Essential only (faster)
python scripts/ingest_all.py --essential
```

**Process:**
1. Fetch from The Odds API → `standardize_game_data()` → Save to `data/raw/the_odds/`
2. Fetch from API-Basketball → `standardize_game_data()` → Save to `data/raw/api_basketball/`
3. Fetch injuries → Standardize → Save to `data/processed/injuries.csv`
4. Fetch betting splits → Standardize → Save to `data/processed/betting_splits.csv`

**Output:**
- Raw JSON files in `data/raw/{source}/` (timestamped)
- Processed CSVs in `data/processed/`

---

### Step 2: Process Raw Data

**Script:** `scripts/build_training_dataset.py`

**Process:**
1. Load standardized odds from `odds_the_odds.csv`
2. Load game outcomes from `game_outcomes.csv`
3. Link odds to outcomes by:
   - Date + home_team + away_team
   - Fuzzy date matching (±1 day tolerance)
4. Build features (team form, rest days, head-to-head, etc.)
5. Merge with betting splits and injuries
6. Merge consensus FG/1H lines from `data/processed/betting_lines.csv`
7. Save to `training_data.csv`

**Output:** `data/processed/training_data.csv`

---

### Step 2b: First-Quarter Feature Dataset (Deprecated)

Q1 training data generation is deprecated. Production markets are 1H + FG only.

---

### Step 3: Feature Engineering

**Module:** `src/modeling/features.py`

**Features Built:**
- Team rolling statistics (PPG, PAPG, FG%, etc.)
- Rest days (home/away team)
- Head-to-head history
- ELO ratings
- Injury impact scores
- Reverse line movement (RLM) signals
- Travel distances (for away teams)

---

## Data Storage

### Directory Structure

```
data/
├── raw/                          # Raw JSON from APIs
│   ├── the_odds/
│   │   └── odds_20251217.json
│   ├── api_basketball/
│   │   ├── teams_20251217.json
│   │   ├── games_20251217.json
│   │   └── statistics_20251217.json
│   └── espn/
│       └── schedule_20251217.json
│
└── processed/                    # Processed CSVs
    ├── odds_the_odds.csv        # Standardized odds
    ├── game_outcomes.csv        # Standardized game results
    ├── betting_splits.csv       # Public betting percentages
    ├── injuries.csv             # Player injuries
    └── training_data.csv        # Final training dataset
```

---

## Data Quality Checks

### At Ingestion

1. **Team Name Validation:**
   - Must normalize to ESPN format
   - Invalid names → ERROR log + skip game

2. **Date Validation:**
   - Must be parseable ISO date
   - Invalid dates → ERROR log + skip game

3. **Required Fields:**
   - `home_team`, `away_team`, `date` required
   - Missing fields → ERROR log + skip game

### At Processing

1. **Linkage Validation:**
   - Odds must match outcomes (date + teams)
   - Unmatched games logged as warnings

2. **Feature Completeness:**
   - Rolling stats require minimum historical games (30+)
   - Missing features → NaN (handled by models)

---

## Error Handling

### Retry Logic

All API calls use `tenacity` retry decorator:
- **Max attempts:** 3
- **Backoff:** Exponential (1s → 8s)

### Logging

**Levels:**
- **ERROR:** Invalid data, standardization failures
- **WARNING:** Missing optional fields, unmatched games
- **INFO:** Successful ingestion, record counts
- **DEBUG:** API request/response details

**Format:** JSON logging with timestamps

---

## Production Considerations

### Rate Limiting

- The Odds API: ~500 requests/month (free tier)
- API-Basketball: Rate limits vary by plan
- Implement delays between batch requests

### Data Freshness

- **Odds:** Update hourly (or more frequently on game days)
- **Game Outcomes:** Update after games complete
- **Injuries:** Update daily
- **Statistics:** Update after each game

### Monitoring

Track:
- Invalid data rate (% of games skipped)
- API failures (retry success rate)
- Data completeness (% of games with all features)
- Standardization errors (team name mismatches)

---

## Commands Reference

### Ingestion

```bash
# Full ingestion (all sources, all tiers)
python scripts/ingest_all.py

# Essential only (faster)
python scripts/ingest_all.py --essential

# Individual sources
python scripts/collect_the_odds.py
python scripts/collect_api_basketball.py
```

### Processing

```bash
# Build training dataset
python scripts/build_training_dataset.py

# Build complete training data (from multiple sources)
python scripts/build_complete_training_data.py

# Import Kaggle data (historical)
python scripts/import_kaggle_betting_data.py
```

### Validation

```bash
# Validate production readiness
python scripts/validate_production_readiness.py

# Check data quality
python scripts/check_data_quality.py

# Reconcile team names
python scripts/reconcile_team_names.py
```

---

## Troubleshooting

### Common Issues

1. **"Failed to normalize team name"**
   - **Cause:** Team name not in mapping/ESPN set
   - **Fix:** Add to `TEAM_NAME_MAPPING` in `standardize.py`

2. **"0 predictions in backtest"**
   - **Cause:** Missing betting lines or insufficient historical data
   - **Fix:** Ensure `spread_line` and `total_line` in training data

3. **"Invalid game data" warnings**
   - **Cause:** Team name standardization failed
   - **Fix:** Check ERROR logs for specific team names, add mappings

4. **API rate limit errors**
   - **Cause:** Too many requests
   - **Fix:** Implement delays, use essential mode, upgrade API plan

---

## Future Enhancements

1. **Real-time Ingestion:** WebSocket subscriptions for live odds
2. **Data Versioning:** Track data lineage and versions
3. **Automated Validation:** CI/CD checks for data quality
4. **Incremental Updates:** Only fetch new/changed data
5. **Multi-season Backfill:** Bulk import historical data

---

## References

- **Standardization Module:** `src/ingestion/standardize.py`
- **The Odds API Client:** `src/ingestion/the_odds.py`
- **API-Basketball Client:** `src/ingestion/api_basketball.py`
- **Dataset Builder:** `src/modeling/dataset.py`
- **Main Ingestion Script:** `scripts/ingest_all.py`

