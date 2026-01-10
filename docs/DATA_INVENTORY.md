# NBA Data Inventory

**Last Updated**: 2026-01-10

## Summary

| Data Type | Source | Date Range | Coverage | Status |
|-----------|--------|------------|----------|--------|
| Games & Scores | Kaggle nba_2008-2025 | 2007-10-30 to 2025-06-22 | 23,118 games | **COMPLETE** |
| FG Betting Lines | Kaggle + TheOdds | 2007+ | 100% | **COMPLETE** |
| FG Betting Lines (2025-26) | TheOdds API | 2025-10-21 to 2026-01-20 | 578 games | **COMPLETE** |
| 1H Betting Lines | TheOdds API | 2023-05 to 2024-06 | 146,348 rows | **COMPLETE** |
| Box Scores (historic) | wyattowalsh/basketball | 1946 to 2023-06-12 | 65,698 games | **COMPLETE** |
| Box Scores (2023-24) | nba_api (NBA.com) | 2023-10 to 2024-06 | 1,230 games | **COMPLETE** |
| Box Scores (2024-25) | nba_api (NBA.com) | 2024-10 to 2025-06 | 1,230 games | **COMPLETE** |
| Box Scores (2025-26) | nba_api (NBA.com) | 2025-10 to 2026-01-10 | 567 games | **COMPLETE** |
| ELO Ratings | Computed from results | All games | 100% | **COMPLETE** |
| Line Movement | TheOdds exports | 2023-10+ | 77% | **COMPLETE** |

---

## Data Sources

### 1. Kaggle: nba_2008-2025.csv
**Location**: `data/external/kaggle/nba_2008-2025.csv`

| Field | Description |
|-------|-------------|
| date, home_team, away_team | Game identification |
| home_score, away_score | Final scores |
| q1_home_score through q4_home_score | Quarter scores |
| home_spread_line, away_spread_line | Pre-game spread |
| total_line | Pre-game over/under |
| home_moneyline, away_moneyline | Moneyline odds |

**Date Range**: 2007-10-30 to 2025-06-22

### 2. TheOdds API Derived Lines
**Location**: `data/historical/derived/theodds_lines.csv`

| Field | Description |
|-------|-------------|
| commence_time | Game start time (UTC) |
| home_team, away_team | Teams |
| fg_spread_*, fg_total_*, fg_ml_* | Full game lines |
| fh_spread_*, fh_total_* | First half lines |
| q1_spread_*, q1_total_* | First quarter lines |

**Date Range**: 2023-10-24 to 2025-06-23

### 3. TheOdds API 1H Exports
**Location**: `data/historical/exports/*_odds_1h.csv`

Explicit first-half odds exports for individual bookmakers.

**Date Range**: May 2023+

### 4. TheOdds API Featured Exports (Line Movement)
**Location**: `data/historical/exports/*_odds_featured.csv`

Opening and closing lines for calculating line movement.

**Date Range**: 2023-10+

### 5. wyattowalsh/basketball (KaggleHub)
**Location**: `data/external/nba_database/`

| File | Description |
|------|-------------|
| game.csv | Game-level box scores |
| line_score.csv | Quarter-by-quarter scores |
| team_game_log.csv | Team game logs |

**Date Range**: 1946-11-01 to 2023-06-12

### 6. NBA.com API (nba_api)
**Location**: `data/raw/nba_api/`

| File | Description |
|------|-------------|
| box_scores_2023_24.csv | 2023-24 season team box scores |
| box_scores_2024_25.csv | 2024-25 season team box scores |
| box_scores_2025_26.csv | 2025-26 season team box scores (current) |

**Status**: COMPLETE - fetched via `scripts/fetch_nba_box_scores.py`

### 7. TheOdds API 2025-26 Season (PREMIER SUBSCRIPTION)
**Location**: `data/historical/the_odds/2025-2026/`

| File | Description |
|------|-------------|
| 2025-2026_all_markets.csv | ALL markets: FG, 1H, Q1, alternates (576 games) |
| 2025-2026_all_markets.json | Raw JSON with all bookmaker data |
| 2025-2026_odds_fg.csv | Full game h2h/spreads/totals (578 games) |

**Market Coverage**:
- Full Game (h2h, spreads, totals): 576 games (100%)
- First Half (h2h_h1, spreads_h1, totals_h1): 495 games (85.9%)
- First Quarter (h2h_q1, spreads_q1, totals_q1): 494 games (85.8%)
- Alternate Lines: 576 games (100%)

**Date Range**: 2025-10-22 to 2026-01-20

**Bookmakers**: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Bovada, and 5 more

---

## Feature Coverage for Backtesting (2023-01-01 to present)

| Feature Category | Coverage | Notes |
|------------------|----------|-------|
| Game scores (FG, Q1-Q4) | 100% | Kaggle source |
| Full game spread/total | 100% | Kaggle + TheOdds |
| 1H spread/total | 77% | TheOdds only (from May 2023) |
| Q1 spread/total | 77% | TheOdds only (from May 2023) |
| Line movement | 77% | Featured exports |
| Box scores (FG%, REB, AST, TOV) | 100% | nba_api fetch COMPLETE |
| ELO ratings | 100% | Computed from game results |
| Rolling stats (PPG, PAPG, etc.) | 100% | Computed from scores |
| Rest days / B2B | 100% | Computed from schedule |

---

## GitHub Resources (Referenced but not yet integrated)

### NBA-Machine-Learning-Sports-Betting
- URL: github.com/... 
- Features: Neural networks, XGBoost, odds integration, bankroll management
- Potential value: Model architectures, evaluation strategies

### NBA-DATASET
- URL: github.com/...
- Features: LeagueSchedule24_25.csv, LeagueSchedule25_26.csv
- Potential value: Schedule data for 2024-25 and 2025-26 seasons

### nba-betting-model
- URL: github.com/...
- Features: XGBoost, Ridge Regression, Streamlit interface
- Potential value: Alternative modeling approaches

---

## Data Quality Notes

1. **Team Name Standardization**: All data sources use `src/data/standardization.py` for consistent team names
2. **Timezone**: All dates converted to CST (America/Chicago)
3. **Match Keys**: Format `YYYY-MM-DD_home_away` for joining across sources
4. **API Limits**: 
   - TheOdds API: Usage-based (we have subscription)
   - nba_api: Free, rate-limited (0.6s delay between calls)
