# Team Name Formats & Standardization Documentation

## Overview

This document catalogs the exact team name formats from each data source and how they're standardized to ESPN format.

**Standard Output Format:** ESPN full team names (e.g., "Los Angeles Lakers", "Boston Celtics")
**Internal Format:** Canonical IDs (e.g., "nba_lal", "nba_bos")

---

## Source Team Name Formats

### 1. API-Basketball (https://v1.basketball.api-sports.io)

#### Endpoint: `/v1/teams`
**Structure:** Flat object
```json
{
  "id": 19,
  "name": "Lakers",
  "code": "LAL",
  "country": {...},
  "logo": "..."
}
```

**Team Name Field:** `name`
**Example Values:**
- "Lakers"
- "Celtics"
- "Warriors"
- "Nets"
- "Heat"
- "76ers"
- "Clippers"
- etc.

**Format:** Short names (usually city/mascot only)

---

#### Endpoint: `/v1/games`
**Structure:** Nested object
```json
{
  "id": 12345,
  "date": "2025-12-18T00:00:00+00:00",
  "teams": {
    "home": {
      "id": 19,
      "name": "Lakers",
      "code": "LAL"
    },
    "away": {
      "id": 14,
      "name": "Celtics",
      "code": "BOS"
    }
  }
}
```

**Home Team Field:** `teams.home.name`
**Away Team Field:** `teams.away.name`

**Format:** Short names (nested in teams object)

**Example Values Found:**
- Home: "Lakers", "Celtics", "Warriors", "Nets", "Heat", "76ers", "Clippers", "Nuggets", "Thunder", "Pelicans", "Spurs", "Trail Blazers", "Suns", "Hawks", "Pistons", "Cavaliers", "Bulls", "Pacers", "Mavericks", "Timberwolves", "Jazz", "Kings", "Magic", "Raptors", "Rockets", "Wizards", "Grizzlies", "Bucks", "Hornets"
- Away: Same format

---

### 2. The Odds API (https://api.the-odds-api.com)

#### Endpoint: `/v4/sports/basketball_nba/participants`
**Structure:** Array of participant objects
```json
[
  {
    "id": "abc123",
    "name": "Los Angeles Lakers",
    "key": "losangeleslakers"
  }
]
```

**Team Name Field:** `name` (or `team` or `id` as fallback)

**Example Values:**
- "Los Angeles Lakers"
- "Boston Celtics"
- "Golden State Warriors"
- "Philadelphia 76ers"
- "LA Clippers" (note: sometimes "LA" not "Los Angeles")
- etc.

**Format:** Full names (usually "City Team")

---

#### Endpoint: `/v4/sports/basketball_nba/events`
**Structure:** Flat object
```json
{
  "id": "abc123",
  "sport_key": "basketball_nba",
  "commence_time": "2025-12-18T00:00:00Z",
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics"
}
```

**Home Team Field:** `home_team` (string)
**Away Team Field:** `away_team` (string)

**Format:** Full names as flat strings

**Example Values:**
- Home: "Los Angeles Lakers", "Boston Celtics", "Golden State Warriors", etc.
- Away: Same format

---

#### Endpoint: `/v4/sports/basketball_nba/odds`
**Structure:** Flat object with bookmakers array
```json
{
  "id": "abc123",
  "sport_key": "basketball_nba",
  "commence_time": "2025-12-18T00:00:00Z",
  "home_team": "Los Angeles Lakers",
  "away_team": "Boston Celtics",
  "bookmakers": [...]
}
```

**Home Team Field:** `home_team` (string)
**Away Team Field:** `away_team` (string)

**Format:** Full names as flat strings (same as events)

**Example Values:** Same as `/events` endpoint

---

#### Endpoint: `/v4/sports/basketball_nba/events/{eventId}/odds`
**Structure:** Same as `/odds` endpoint

**Home/Away Fields:** Same as `/odds` endpoint

**Format:** Full names as flat strings

---

### 3. Action Network (Betting Splits)

**Format:** Various (HTML scraping)
**Home/Away Fields:** Extracted from HTML structure

**Note:** Team names extracted from HTML and standardized to ESPN format.

---

## Standardization Flow

### Input Formats Summary

| Source | Home Team Field | Away Team Field | Format Type | Example Values |
|--------|----------------|-----------------|-------------|----------------|
| API-Basketball `/teams` | N/A | N/A | Flat: `name` | "Lakers", "Celtics" |
| API-Basketball `/games` | `teams.home.name` | `teams.away.name` | Nested | "Lakers", "Celtics" |
| The Odds API `/events` | `home_team` | `away_team` | Flat string | "Los Angeles Lakers" |
| The Odds API `/odds` | `home_team` | `away_team` | Flat string | "Los Angeles Lakers" |
| The Odds API `/participants` | N/A | N/A | Flat: `name` | "Los Angeles Lakers" |

### Standardization Process

1. **Extract team names** from source-specific fields
2. **Normalize to ESPN format** using `normalize_team_to_espn()`
3. **Validate** - invalid names set to empty string (game skipped)

### Master Database: `team_mapping.json`

Canonical IDs → Variants mapping:
```json
{
  "nba_lal": ["los angeles lakers", "lakers", "la lakers", "lal"],
  "nba_bos": ["boston celtics", "celtics", "bos"],
  ...
}
```

**Canonical IDs → ESPN Names:**
```python
"nba_lal" → "Los Angeles Lakers"
"nba_bos" → "Boston Celtics"
```

---

## Team Variant Coverage

### API-Basketball Variants (Short Names)
- "Lakers", "Celtics", "Warriors", "Nets", "Heat"
- "76ers", "Sixers", "Clippers", "Nuggets", "Thunder"
- "Pelicans", "Spurs", "Trail Blazers", "Blazers", "Suns"
- "Hawks", "Pistons", "Cavaliers", "Cavs", "Bulls"
- "Pacers", "Mavericks", "Mavs", "Timberwolves", "Twolves"
- "Jazz", "Kings", "Magic", "Raptors", "Rockets", "Wizards"
- "Grizzlies", "Bucks", "Hornets"

### The Odds API Variants (Full Names)
- "Los Angeles Lakers", "Boston Celtics"
- "Golden State Warriors", "Philadelphia 76ers"
- "LA Clippers" or "Los Angeles Clippers"
- "New York Knicks", "Brooklyn Nets"
- "Miami Heat", etc.

### ESPN Standard Output (All Sources)
All standardized to:
- "Los Angeles Lakers" (not "Lakers" or "LAL")
- "Boston Celtics" (not "Celtics" or "BOS")
- "Golden State Warriors" (not "Warriors" or "GSW")
- etc.

---

## Running Team Name Diagnostic

**To see actual formats from all sources:**

```bash
# In container
docker compose -f docker-compose.backtest.yml run --rm backtest-shell

# Run diagnostic
python scripts/diagnose_team_names.py

# Or via entrypoint
docker compose -f docker-compose.backtest.yml run --rm backtest-shell diagnose-team-names
```

**Output:**
- `data/diagnostics/team_name_diagnostic_*.md` - Markdown report
- `data/diagnostics/team_name_diagnostic_*.json` - Raw data

---

## Standardization Validation

**All team names MUST:**
1. Extract from source-specific field structure
2. Normalize via `normalize_team_to_espn()`
3. Validate (`is_valid=True`)
4. Use ESPN full name format

**Invalid data handling:**
- `is_valid=False` → Team name set to empty string
- Game marked `_data_valid=False`
- Game skipped in pipeline (prevents fake data)

---

## Format Consistency Rules

**All sources → ESPN format:**
- ✅ Full team names (not abbreviations)
- ✅ City + Team name format
- ✅ Consistent capitalization
- ✅ "Los Angeles Lakers" (not "LA Lakers" or "Lakers")
- ✅ "Philadelphia 76ers" (not "76ers" or "Sixers")

**Internal processing:**
- Uses canonical IDs (nba_lal, nba_bos) for matching
- Outputs ESPN full names for display/data

---

## Adding New Team Name Variants

If a variant fails standardization:

1. **Add to `src/ingestion/team_mapping.json`:**
```json
{
  "nba_lal": [
    "los angeles lakers",
    "lakers",
    "la lakers",
    "NEW_VARIANT_HERE"  // Add lowercase variant
  ]
}
```

2. **Run diagnostic to verify:**
```bash
python scripts/diagnose_team_names.py
```

3. **Check report** for any remaining failures

---

## Quick Reference: Field Extraction

### API-Basketball Games
```python
home_team = game["teams"]["home"]["name"]  # "Lakers"
away_team = game["teams"]["away"]["name"]  # "Celtics"
```

### The Odds API Events/Odds
```python
home_team = event["home_team"]  # "Los Angeles Lakers"
away_team = event["away_team"]  # "Boston Celtics"
```

### After Standardization
```python
home_team, home_valid = normalize_team_to_espn(home_team_raw, source="api_basketball")
away_team, away_valid = normalize_team_to_espn(away_team_raw, source="the_odds")

# Result: "Los Angeles Lakers", "Boston Celtics" (ESPN format)
```

---

**This documentation is generated/updated by running `scripts/diagnose_team_names.py`**
