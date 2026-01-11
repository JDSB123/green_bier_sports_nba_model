# NBA Data Inventory

**Last Updated**: 2026-01-10

## Summary

| Data Type | Source | Date Range | Coverage | Status |
|-----------|--------|------------|----------|--------|
| Games & Scores | Kaggle nba_2008-2025 | 2007-10-30 to 2025-06-22 | 23,118 games | **COMPLETE** |
| FG Betting Lines | Kaggle + TheOdds | 2007+ | 100% | **COMPLETE** |
| 1H Betting Lines | TheOdds API | 2023-05+ | 78.4% | **COMPLETE** |
| Box Scores (historic) | wyattowalsh/basketball | 1946 to 2023-06-12 | 65,698 games | **COMPLETE** |
| Box Scores (2023-26) | nba_api (NBA.com) | 2023-10 to present | 2,990 games | **COMPLETE** |
| ELO Ratings | Computed from results | All games | 100% | **COMPLETE** |
| Line Movement | TheOdds exports | 2023-10+ | 66% | **COMPLETE** |

---

## Final Training Data

### PRIMARY: `data/processed/training_data_all_seasons.csv`

**THE single source of truth for all training and backtesting.**

| Metric | Value |
|--------|-------|
| **Games** | 4,456 |
| **Date Range** | 2023-01-01 to 2026-01-19 |
| **Columns** | 270+ |

**Coverage**:
- FG spread/total: ~99%
- 1H spread/total: ~70%
- Box scores: 100%
- ELO ratings: 100%
- Rolling stats: 100%
- Line movement: 66.2%

**Seasons Included** (NBA seasons run Oct-Jun):
- 2022-23: 774 games
- 2023-24: 1,319 games
- 2024-25: 1,321 games
- 2025-26: 1,042 games (in progress)

### LEGACY: `data/processed/training_data_complete_2023.csv`

Original merged file (3,979 games). Superseded by `training_data_all_seasons.csv`.

### SECONDARY: `data/processed/training_data_2025_26.csv`

2025-26 season focused file with box score integration (576 games, 99.3% box score coverage).

---

## Data Storage Structure

```
data/
├── processed/                    # FINAL OUTPUT
│   ├── training_data_complete_2023.csv   # ← USE THIS
│   ├── training_data_2025_26.csv         # 2025-26 focused
│   ├── data_manifest.json
│   └── models/                   # Trained model artifacts
│       └── model_pack.json
│
├── raw/                          # API fetches
│   ├── nba_api/
│   │   ├── box_scores_2023_24.csv  (2,386 rows)
│   │   ├── box_scores_2024_25.csv  (2,460 rows)
│   │   └── box_scores_2025_26.csv  (1,134 rows)
│   └── github/
│       └── fivethirtyeight_elo.csv
│
├── external/                     # Third-party datasets
│   └── kaggle/
│       └── nba_2008-2025.csv     # Primary scores/lines source
│
├── historical/                   # TheOdds API historical data
│   ├── derived/
│   │   └── theodds_lines.csv     # Aggregated FG/1H lines 2021-2025
│   ├── exports/
│   │   ├── 2023-2024_odds_1h.csv    (86,512 rows)
│   │   ├── 2024-2025_odds_1h.csv    (59,836 rows)
│   │   ├── 2023-2024_odds_featured.csv  (line movement)
│   │   └── 2024-2025_odds_featured.csv
│   ├── the_odds/
│   │   ├── 2025-2026/
│   │   │   ├── 2025-2026_all_markets.csv   # FG/1H/Q1/alts
│   │   │   └── 2025-2026_all_markets.json  # Raw bookmaker data
│   │   ├── events/        (859 JSON files)
│   │   ├── odds/          (773 JSON files)
│   │   └── metadata/
│   └── elo/
│       └── fivethirtyeight_elo_historical.csv
│
└── backtest_results/             # Backtest outputs
    └── *.csv, *.json
```

---

## Data Sources

### 1. Kaggle: nba_2008-2025.csv
**Location**: `data/external/kaggle/nba_2008-2025.csv`

| Field | Description |
|-------|-------------|
| date, home_team, away_team | Game identification |
| score_home, score_away | Final scores |
| q1_home through q4_home | Quarter scores |
| spread, total | Pre-game lines |
| moneyline_home, moneyline_away | Moneyline odds |

**Date Range**: 2007-10-30 to 2025-06-22 (23,118 games)

### 2. TheOdds API (PREMIER SUBSCRIPTION)
**Location**: `data/historical/the_odds/2025-2026/`

| File | Description |
|------|-------------|
| 2025-2026_all_markets.csv | ALL markets: FG, 1H, Q1, alternates |
| 2025-2026_all_markets.json | Raw JSON with all bookmaker data |

**Market Coverage** (2025-26 season):
- Full Game (h2h, spreads, totals): 576 games (100%)
- First Half (h2h_h1, spreads_h1, totals_h1): 495 games (86%)
- First Quarter (h2h_q1, spreads_q1, totals_q1): 494 games (86%)
- Alternate Lines: 576 games (100%)

**Bookmakers**: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Bovada, +5 more

### 3. NBA.com API (nba_api)
**Location**: `data/raw/nba_api/`

| File | Games | Description |
|------|-------|-------------|
| box_scores_2023_24.csv | 1,193 | Full team box scores |
| box_scores_2024_25.csv | 1,230 | Full team box scores |
| box_scores_2025_26.csv | 567 | Full team box scores (current) |

**Stats Included**: FGM/FGA, FG3M/FG3A, FTM/FTA, OREB/DREB, AST, STL, BLK, TO, PF, PTS

### 4. wyattowalsh/basketball (KaggleHub)
**Location**: `data/external/nba_database/` (if downloaded)

Historical box scores from 1946 to 2023-06-12 (65,642 games).

---

## Feature Categories in Training Data

| Category | Columns | Description |
|----------|---------|-------------|
| Core | 5 | match_key, game_date, home/away_team, season |
| FG Scores | 20 | home/away_score, margins, Q1-Q4 |
| FG Lines | 15 | Kaggle + TheOdds spreads/totals/MLs |
| 1H Lines | 15 | TheOdds 1H spreads/totals/MLs |
| Box Score | 33 | eFG%, 3P rate, FT rate, TOV%, OREB%, Off/Def/Net Rating |
| ELO | 4 | home_elo, away_elo, elo_diff, elo_prob |
| Rolling Stats | 126 | 5/10/20-game windows for all metrics |
| Labels | 6 | fg/1h spread_covered, total_over, home_win |

---

## Data Quality

1. **Team Name Standardization**: All sources use `src/data/standardization.py`
2. **Timezone**: All dates in CST (America/Chicago)
3. **Match Keys**: Format `YYYY-MM-DD_home_away` for cross-source joins
4. **No Data Leakage**: Features computed using only prior games
5. **API Rate Limiting**: Implemented for nba_api (0.6s delay)
