# Data Single Source of Truth

**Last Updated:** 2026-01-12

This document defines the authoritative data sources and pipeline for the NBA prediction system.

---

## ⚠️ CRITICAL: Azure Blob is the ONLY Source

**ALL historical data lives in Azure Blob Storage. NOT in git. NOT locally.**

- ❌ Do NOT commit data files to git
- ❌ Do NOT store historical data locally long-term
- ❌ Do NOT clone/pull data from git
- ✅ ALWAYS pull from Azure Blob via `--sync-from-azure` (now default)

---

## 🔒 Azure Blob Storage (CANONICAL SOURCE)

Historical data is stored in Azure Blob Storage as the **single source of truth**.

### Location
```
Storage Account: nbagbsvstrg
Container:       nbahistoricaldata
```

### Versioned Data
| Path | Description |
|------|-------------|
| `training_data/v2026.01.11/` | Version-stamped release |
| `training_data/latest/` | Always points to latest validated version |

### Quality Gates (enforced before upload)
- ✅ Minimum 3,000 games
- ✅ Minimum 50 features/columns
- ✅ 80%+ injury coverage
- ✅ 90%+ odds coverage
- ✅ No nulls in critical columns (game_id, teams, scores)
- ✅ Score ranges validated (50-200)
- ✅ SHA256 checksum in manifest

### Scripts
```bash
# Upload quality-checked data to Azure
python scripts/upload_training_data_to_azure.py --force --version v2026.01.XX

# Download from Azure (single source of truth)
python scripts/download_training_data_from_azure.py --version latest --verify

# List available versions
python scripts/download_training_data_from_azure.py --list
```

---

## Training Data

### Master File (Local)
```
data/processed/training_data_complete_2023_with_injuries.csv
```

**Date Range:** 2023-01-01 to 2026-01-09
**Games:** 3,969
**Columns:** 327
**Injury Coverage:** 100% (via Kaggle inference)
**Odds Coverage:** 100%
**Model Features:** 55/55 (100% coverage)

### How to Rebuild

```bash
# Build training data (automatically syncs from Azure Blob)
python scripts/build_training_data_complete.py

# For local dev with pre-cached data only (NOT recommended for production)
python scripts/build_training_data_complete.py --no-azure-sync
```

This master script:
1. **Syncs from Azure Blob** (default behavior - pulls all historical data)
2. Merges ALL data sources (see below)
3. Computes ELO ratings, rolling stats, situational features
4. Runs `fix_training_data_gaps.py` automatically
5. Runs `complete_training_features.py` automatically
6. Outputs canonical file: `data/processed/training_data.csv`

**NEVER** run fix scripts or feature scripts independently during normal operations.
**NEVER** assume local data is fresh - always sync from Azure.

---

## Data Sources

### 1. Kaggle (nba_2008-2025.csv)
**Location:** Azure Blob (`nbagbsvstrg/nbahistoricaldata/historical/exports/`) and local cache when pulled
**Coverage:** 2008-2025 seasons
**Contents:**
- Final scores (home_score, away_score)
- Quarter scores (q1-q4)
- Betting lines (spread, total, moneyline)
- First half lines (1h_spread, 1h_total)

### 2. TheOdds API
**Location:** Azure Blob (`nbagbsvstrg/nbahistoricaldata/historical/the_odds/` and `historical/exports/`); pull to temp for use
**Coverage:** 2021-present (Comprehensive for 2023+)
**Contents:**
- Full game lines (FG spread, total, moneyline)
- First half lines (1H spread, total, moneyline)
- Line movement data
- Per-bookmaker odds

### 3. nba_database (wyattowalsh/basketball)
**Location:** Azure Blob (`nbagbsvstrg/nbahistoricaldata/historical/exports/`) and temp local cache when needed
**Coverage:** 1946-2023 (historical), ongoing updates
**Files:**
| File | Records | Contents |
|------|---------|----------|
| `game.csv` | 65K+ | Box scores (FGA, FTA, OREB, TOV, etc.) |
| `inactive_players.csv` | 110K | Who was inactive per game |
| `line_score.csv` | 65K+ | Period-by-period scores |
| `common_player_info.csv` | 3,632 | Player metadata (no stats) |

### 4. NBA API (nba_api)
**Location:** Generated locally per run (not stored long term); upload derived outputs to Azure if persisted
**Coverage:** 2023-present
**Contents:**
- Box scores (per season)
- Quarter scores (2025-26)

### 5. FiveThirtyEight ELO
**Location:** Azure Blob (`nbagbsvstrg/nbahistoricaldata/historical/elo/`) and temp local cache when needed
**Coverage:** Historical
**Contents:**
- Team ELO ratings

### 6. Kaggle eoinamoore (Player Box Scores)
**Location:** Azure Blob (`nbagbsvstrg/nbahistoricaldata/historical/exports/`) and temp local cache when needed
**Dataset:** `eoinamoore/historical-nba-data-and-player-box-scores`
**Coverage:** 1947-present (updated daily!)
**Files:**
| File | Size | Contents |
|------|------|----------|
| `PlayerStatistics.csv` | 305 MB | Player box scores (1.6M records) |
| `Players.csv` | 0.5 MB | Player biographical info (6,681 players) |
| `Games.csv` | 10 MB | All NBA games (72K+) |
| `LeagueSchedule24_25.csv` | 0.1 MB | Current season schedule |

**Purpose:** Infer inactive players by comparing "who played" vs "who should have played"

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
| **Injury Impact** | **95.4%** | nba_database + Kaggle-inferred |

### Injury Impact Calculation

**Data Sources (Combined):**
1. `data/external/nba_database/inactive_players.csv` - 110K records (through June 2023)
2. `data/processed/inactive_players_kaggle_supplement.csv` - 19K records (July 2023-present, inferred from Kaggle box scores)

**Coverage by Year:**
| Year | Games | With Injury Data |
|------|-------|------------------|
| 2023 | 1,255 | 92.9% |
| 2024 | 1,323 | 98.1% |
| 2025 | 1,330 | 95.0% |
| 2026 | 61 | 98.4% |

**Impact Scoring:**
- PPG component (max 5 pts): 20+ PPG = 5 points
- MPG component (max 2 pts): 30+ MPG = 2 points
- Draft component (max 3 pts): 1st round = 2.0, Top 5 = 2.5, #1 overall = 3.0

**Pipeline:**
```bash
# Download Kaggle box scores (daily-updated)
python scripts/download_kaggle_player_data.py

# Infer inactive players from box scores
python scripts/download_kaggle_player_data.py

# Merge all sources and update training data
python scripts/build_training_data_complete.py
```

**Output:**
- `data/processed/inactive_players_merged.csv` - Combined inactive players
- `data/processed/training_data_complete_2023_with_injuries.csv` - Training data with injury features

### Real-Time Only (Defaults for Historical)
| Feature | Status | Notes |
|---------|--------|-------|
| Betting splits | Defaults | Real-time only, set to neutral |

---

## Known Gaps & Limitations

### 1. Moneylines (Partial Coverage ~70%)
- **Spreads & Totals:** ✅ **100% Coverage** (Full Game) for 2023-Present.
- **1st Half Markets:** ✅ **98%+ Coverage** since 2023-24 Season. (Jan-June 2023 has gaps).
- **Moneylines:** ~70% coverage (TheOdds API has gaps in ML history).
- **Impact:** Models rely primarily on Spread/Total data which is complete.

### 2. Player Impact/Injuries (RESOLVED)
- **Status:** ✅ **100% Coverage** for 2023-Present.
- **Method:** `inactive_players_kaggle_supplement.csv` infers inactives from daily box scores.
- **Feature:** `home_injury_impact` / `away_injury_impact` fully populated.

### 3. Pace Features
- `game.csv` has possessions data (FGA, FTA, OREB, TOV)
- **Computed:** Using league average (100) as baseline
- **Improvement potential:** Compute rolling pace per team

### 4. Travel Features (RESOLVED)
- **Status:** ✅ **100% Coverage**.
- **Features:** `away_travel_distance`, `away_timezone_change`, `is_away_long_trip` fully populated.
- **Method:** `team_factors.py` correctly calculates distance between stadiums.

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
python scripts/fetch_nba_box_scores.py  # Update box scores
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
1. Check `data/processed/training_data.csv` modification date
2. Rebuild if needed: `python scripts/build_training_data_complete.py`

---

## Archive Policy (Azure Blob Storage)

- Archives live in Azure Blob (`nbagbsvstrg`/`nbahistoricaldata/archives/`), not in git.
- Local archive folder is gitignored; use it only as a temp staging area.
- Upload archives with `.\scripts\sync_archives_to_azure.ps1`; add `-Cleanup` to delete local after upload.
- Download if needed:
```bash
az storage blob download-batch \
  --account-name nbagbsvstrg \
  --source nbahistoricaldata \
  --destination . \
  --pattern 'archives/*'
```
Goal: Azure is the single source; local copies are temporary.
