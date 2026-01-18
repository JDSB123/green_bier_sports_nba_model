# Data Single Source of Truth

Last updated: 2026-01-16

This document defines THE ONE authoritative training data file.

---

## THE CANONICAL FILE

```
data/processed/training_data.csv
```

**This is THE ONLY training data file. There is no other.**

| Attribute | Value |
|-----------|-------|
| Rows | 3,969 games |
| Columns | 327 columns |
| Date Range | 2023-01-01 to 2026-01-08 |
| Injury Data | ✅ 100% coverage |
| FG Labels | ✅ 100% coverage |
| 1H Labels | 81.2% overall coverage (99.6% since 2023-05-01) |

### Season Breakdown
- Early 2023 segment: drives overall 1H coverage down (historical 1H lines are sparse)
- Since 2023-05-01 window: 99.6%+ 1H coverage (enforced window for coverage gates)

---

## Azure Blob Storage (Mirror)

Storage account: `nbagbsvstrg`
Container: `nbahistoricaldata`

Versioned training data:
- `training_data/vYYYY.MM.DD/`
- `training_data/latest/`

Quality gates (enforced before upload):
- Minimum 3,000 games
- Minimum 50 features/columns
- 80%+ injury coverage
- 90%+ odds coverage
- No nulls in critical columns (game_id, teams, scores)
- Score ranges validated (50-200)
- SHA256 checksum in manifest

Scripts:

```bash
# Upload validated data to Azure
python scripts/upload_training_data_to_azure.py --force --version v2026.01.XX

# Download canonical data
python scripts/download_training_data_from_azure.py --version latest --verify

# List available versions
python scripts/download_training_data_from_azure.py --list
```

---

## Backtest Data Policy (Canonical Only)

Backtests must use the audited canonical dataset:

```
data/processed/training_data.csv
```

Rules:
- Do NOT rebuild or merge raw data for backtests.
- Do NOT use training_data_all_seasons or any ad-hoc merge outputs.
- If the file is missing, download it from Azure (`training_data/latest/`) and verify checksum.

Backtesting phase is consume-only:
- Backtest scripts must only read the audited `training_data.csv` artifact.
- Any rebuild/merge/backfill belongs to the data engineering phase only.

---

## Data Engineering (Rebuilds Are Separate)

Rebuilds are for data engineering only and are not part of backtest runs:

```bash
# Default behavior downloads prebuilt audited data from Azure (no rebuild)
python scripts/build_training_data_complete.py

# Explicit rebuild from raw sources (data engineering only)
python scripts/build_training_data_complete.py --rebuild-from-raw
```

Outputs:
- Canonical file: `data/processed/training_data.csv`

---

## Data Sources (Summary)

1. Kaggle (nba_2008-2025.csv)
   - Scores, quarter scores, betting lines, first-half lines

2. The Odds API
   - Full-game and first-half lines, line movement, bookmaker odds

3. nba_database (wyattowalsh/basketball)
   - Box scores and period-by-period scores

4. NBA API (nba_api)
   - Box scores and quarter scores (2025-26+)

5. FiveThirtyEight ELO
   - Team ELO ratings

6. Kaggle eoinamoore (historical NBA player box scores)
   - Used to infer inactive players

---

## Feature Coverage (Snapshot)

Full coverage (100%):
- FG labels: fg_spread_covered, fg_total_over, fg_home_win
- Scores: home_score, away_score, fg_margin, fg_total_actual
- Injury impact: 7 columns, 100% coverage
- Rest, ELO, derived, rolling, H2H, predicted features

Partial coverage:
- 1H labels: 78% (2022-23 lacks 1H betting lines)
- Moneylines: best available from The Odds API

---

## DEPRECATED FILES (DO NOT USE)

The following files have been removed or deprecated:
- `master_training_data.csv` - DELETED, merged into training_data.csv
- `training_data_canonicalized.csv` - DELETED
- `training_data_2023_with_injuries.csv` - DELETED
- Any file matching `*training*.csv` except `training_data.csv`
