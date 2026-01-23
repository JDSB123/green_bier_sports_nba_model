# Data Inventory (Azure as Source of Truth)

**Last Updated:** 2026-01-12

## Canonical storage
- Historical/backtest data: Azure Blob `nbagbsvstrg` / container `nbahistoricaldata` / prefix `historical/`
- Archived picks & backtest artifacts: same container (`archived_picks/`, `models/backtest/`)
- Live artifacts: ACR `nbagbsacr`, Container App `nba-gbsv-api`

## Training data in use
- File: `data/processed/training_data.csv` (canonical working copy with injuries, committed to git)
- Manifest: `data/processed/training_data_manifest.json`
- Coverage: 3,969 games, 2023-01-01 to 2026-01-09, 327 columns; odds & injury coverage 1.0

## Sources merged
- The Odds API (featured + 1H odds, line movement)
- API-Basketball (box scores/advanced stats/standings)
- Injury feeds (impact scores, star-out flags)
- FiveThirtyEight Elo
- Travel/rest heuristics and betting splits/RLM signals

## Local directories (policy)
- `data/historical/`: TEMP cache only; pull from Azure for a run, then delete. Not in git.
- `data/processed/`: Generated training sets (kept locally, not pushed to Azure automatically).
- `data/backtest_results/`: Backtest outputs (JSON/CSV) from runs.

## Retrieval pattern for historical data
```
az storage blob download-batch --account-name nbagbsvstrg --auth-mode login \
  --source nbahistoricaldata --destination data/historical --pattern "historical/*"
```
Then run backtest or builders; delete `data/historical` afterward to avoid stale data.
