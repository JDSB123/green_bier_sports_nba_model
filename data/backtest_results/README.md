## Backtest results (versioned snapshots)

This folder contains **versioned backtest output artifacts** committed to git so we can
reference historical results later (audit trail).

### Files

- `backtest_results_20260105_131610.json`: initial leakage-audited FG Spread backtest (random forest)
- `backtest_results_20260105_131635.json`: initial leakage-audited run for all markets attempted

### Notes

- These JSON files are **outputs** from `scripts/backtest_market_models.py`.
- Do not overwrite existing snapshot files; create a new timestamped output instead.
- If results are re-generated, keep prior snapshots in git for comparability.

