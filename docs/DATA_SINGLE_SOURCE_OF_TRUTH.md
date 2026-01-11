# Data Single Source of Truth

**Last Updated:** 2026-01-11

This document defines the authoritative data sources and pipeline for the NBA prediction system.

---

## Training Data

### Master File
```
data/processed/training_data_complete_2023.csv
```

**Date Range:** 2023-01-01 to present  
**Games:** 3,969+  
**Columns:** 324  
**Model Features:** 55/55 (100% coverage)

### How to Rebuild

```bash
# ONLY WAY to build training data
python scripts/build_training_data_complete.py --start-date 2023-01-01
```

This master script:
1. Merges ALL data sources (see below)
2. Computes ELO ratings, rolling stats, situational features
3. Runs `fix_training_data_gaps.py` automatically
4. Runs `complete_training_features.py` automatically
5. Outputs complete training file with all 55 model features

**NEVER** run fix scripts or feature scripts independently during normal operations.

---

## Data Sources

### 1. Kaggle (nba_2008-2025.csv)
**Location:** `data/external/kaggle/`  
**Coverage:** 2008-2025 seasons  
**Contents:**
- Final scores (home_score, away_score)
- Quarter scores (q1-q4)
- Betting lines (spread, total, moneyline)
- First half lines (h2_spread, h2_total)

### 2. TheOdds API
**Location:** `data/historical/derived/`, `data/historical/exports/`, `data/historical/the_odds/`  
**Coverage:** 2021-present  
**Contents:**
- Full game lines (FG spread, total, moneyline)
- First half lines (1H spread, total, moneyline)
- Line movement data
- Per-bookmaker odds

### 3. nba_database (wyattowalsh/basketball)
**Location:** `data/external/nba_database/`  
**Coverage:** 1946-2023 (historical), ongoing updates  
**Files:**
| File | Records | Contents |
|------|---------|----------|
| `game.csv` | 65K+ | Box scores (FGA, FTA, OREB, TOV, etc.) |
| `inactive_players.csv` | 110K | Who was inactive per game |
| `line_score.csv` | 65K+ | Period-by-period scores |
| `common_player_info.csv` | 3,632 | Player metadata (no stats) |

### 4. NBA API (nba_api)
**Location:** `data/raw/nba_api/`  
**Coverage:** 2023-present  
**Contents:**
- Box scores (per season)
- Quarter scores (2025-26)

### 5. FiveThirtyEight ELO
**Location:** `data/external/fivethirtyeight/`  
**Coverage:** Historical  
**Contents:**
- Team ELO ratings

---

## Feature Coverage

### Full Coverage (100%)
| Feature Category | Features |
|-----------------|----------|
| **FG Labels** | fg_spread_covered, fg_total_over, fg_home_win |
| **1H Labels** | 1h_spread_covered, 1h_total_over |
| **Scores** | home_score, away_score, fg_margin, fg_total_actual |
| **Basic Stats** | home_ppg, away_ppg, home_win_pct, away_win_pct |
| **Rest** | home_rest_days, away_rest_days, rest_diff |
| **ELO** | home_elo, away_elo, elo_diff |
| **Derived** | ppg_diff, win_pct_diff, net_rating_diff |
| **Rolling** | margin_std, score_std, form_trend |
| **H2H** | h2h_games, h2h_margin |
| **Predicted** | predicted_margin, predicted_total |

### Partial Coverage
| Feature | Coverage | Notes |
|---------|----------|-------|
| Moneylines | 69.5% | Best available from TheOdds |
| Travel features | 0% | team_factors module needs fix |

### Not Available Historically
| Feature | Status | Notes |
|---------|--------|-------|
| Betting splits | Defaults | Real-time only, set to neutral |
| Injury impact (PPG) | Defaults | No player-level stats available |

---

## Known Gaps & Limitations

### 1. Moneylines (69.5% coverage)
- TheOdds has best coverage at 69.5%
- Kaggle has only ~3% for 2023+
- **Cannot improve** without additional data source

### 2. Player Impact/Injuries
- `inactive_players.csv` has 110K records of who was inactive
- **Missing:** Player PPG/stats to calculate impact
- Currently set to neutral defaults (0 impact)

### 3. Pace Features
- `game.csv` has possessions data (FGA, FTA, OREB, TOV)
- **Computed:** Using league average (100) as baseline
- **Improvement potential:** Compute rolling pace per team

### 4. Travel Features
- `team_factors.py` module has distance calculations
- **Currently:** Set to 0 due to import/mapping issues
- **Fix needed:** Update team name mappings

---

## Data Quality Checks

Run validation before training:

```bash
python scripts/validate_training_data.py
```

Checks:
- Label balance (should be ~50/50)
- Feature coverage (all 55 required)
- Date range (2023+)
- No data leakage

---

## Updating Data

### Daily (Production)
```bash
python scripts/collect_the_odds.py  # Get latest odds
python scripts/run_slate.py          # Generate predictions
```

### Weekly (Optional)
```bash
python scripts/fetch_quarter_scores.py  # Update quarter scores
python scripts/fetch_box_scores_parallel.py  # Update box scores
```

### After Season / Major Updates
```bash
python scripts/build_training_data_complete.py  # Rebuild everything
python scripts/train_models.py  # Retrain models
```

---

## Troubleshooting

### "Feature X missing from training data"
Run: `python scripts/complete_training_features.py`

### "Label coverage incomplete"
Run: `python scripts/fix_training_data_gaps.py`

### "Need to rebuild everything"
Run: `python scripts/build_training_data_complete.py --start-date 2023-01-01`

### "Data looks stale"
1. Check `data/processed/training_data_complete_2023.csv` modification date
2. Rebuild if needed: `python scripts/build_training_data_complete.py`
