# Data Single Source of Truth

Last updated: 2026-01-12

This document defines the authoritative data sources and how canonical training data is used.

---

## Canonical Source (Git + Azure Blob)

Canonical training data is committed in git and mirrored to Azure Blob Storage.
Other historical/raw inputs remain in Azure only.

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
- If the file is missing, restore it from git or download from Azure.

---

## Data Engineering (Rebuilds Are Separate)

Rebuilds are for data engineering only and are not part of backtest runs:

```bash
# Rebuild full training data from raw sources
python scripts/build_training_data_complete.py
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
- 1H labels: 1h_spread_covered, 1h_total_over
- Scores: home_score, away_score, fg_margin, fg_total_actual
- Rest, ELO, derived, rolling, H2H, predicted features

Partial coverage:
- Moneylines: best available from The Odds API
- Travel features: not yet implemented
- Injury impact: 95%+ via Kaggle inference
