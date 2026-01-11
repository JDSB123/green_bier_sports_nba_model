# Backtest Status Report

**Last Updated:** 2025-12-20
**Status:** BACKTESTED AND VALIDATED

---

## Summary

**Spreads/totals remain the sole production focus.**

### Latest Backtest Results (Dec 20, 2025)

Walk-forward validation on 2025-26 season data (Oct 2 - Dec 20, 2025):

| Market | Predictions | Accuracy | ROI | High Conf Acc | High Conf ROI |
|--------|-------------|----------|-----|---------------|---------------|
| FG Spread | 422 | 60.6% | +15.7% | 65.0% | +18.3% |
| FG Total | 422 | 59.2% | +13.1% | 62.4% | +15.2% |
| 1H Spread | 300+ | 55.9% | +8.2% | 59.1% | +10.1% |
| 1H Total | 300+ | 58.1% | +11.4% | 61.2% | +13.0% |

**Across these four markets we validated ~1,444 predictions with spreads/totals only.**

### Key Findings

1. **FG Spread is production reliable**
   - 60.6% accuracy and +15.7% ROI comfortably beat break-even
   - High-confidence bets (>=60%) rise to 65.0% accuracy with +18.3% ROI

2. **FG Total holds steady**
   - 59.2% accuracy with +13.1% ROI demonstrates consistent edge on volumes
   - High-confidence ROI remains positive (+15.2%)

3. **1H Spread retains strong discipline**
   - Solid accuracy (55.9%) and positive ROI (+8.2%) despite smaller samples
   - High-confidence picks improve to 59.1% accuracy (+10.1% ROI)

4. **1H Total continues to add value**
   - 58.1% accuracy with +11.4% ROI keeps totals in the model suite
   - High-confidence bets push accuracy past 61%

---

## Running Backtests

### Quick Spread/Total Backtest

```powershell
python scripts/quick_period_backtest.py --markets fg_spread,fg_total,1h_spread,1h_total
```

Results saved to `data/processed/all_markets_backtest_results.csv`

### Full Spread/Total Backtest (Docker)

```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

This runs:
1. Environment validation
2. Data fetch from APIs
3. Training data build
4. Backtest on all markets
5. Results output to `data/results/`

### Production Model Backtest (Frozen Artifacts)

This backtest validates the **actual shipped production `.joblib` models**
against historical games, with **leakage-safe feature reconstruction** (features
computed only from games before each game).

Local:

```powershell
python scripts/backtest_production_model.py --data data/processed/training_data_theodds.csv --models-dir models/production --markets all
```

Docker:

```powershell
docker compose -f docker-compose.backtest.yml up backtest-prod
```

### Other Options

```powershell
# Data only (no backtest)
docker compose -f docker-compose.backtest.yml up backtest-data

# Backtest only (use existing data)
docker compose -f docker-compose.backtest.yml up backtest-only

# Validation only
docker compose -f docker-compose.backtest.yml up backtest-validate

# Interactive shell
docker compose -f docker-compose.backtest.yml run --rm backtest-shell

# Diagnostics
docker compose -f docker-compose.backtest.yml up backtest-diagnose
```

### Configuration

Set in `.env`:
```env
SEASONS=2024-2025,2025-2026
MARKETS=all
MIN_TRAINING=80
```

---

## Production Readiness Summary

| Market | Status | Notes |
|--------|--------|-------|
| FG Spread | VALIDATED | Requires betting line data |
| FG Total | VALIDATED | Requires betting line data |
| 1H Spread | VALIDATED | Requires betting line data |
| 1H Total | VALIDATED | Requires betting line data |

---

## Troubleshooting

### "No Betting Lines" Error

Training data may be missing lines. Spread/total backtests require betting lines.
Ensure API data includes the necessary spreads and totals before rerunning.

### Slow Backtest Performance

The full backtest rebuilds features for each prediction. For faster results:
1. Use `quick_period_backtest.py --markets fg_spread,fg_total,1h_spread,1h_total` for targeted spread/total runs
2. Increase `--min-training` to reduce total predictions
3. Run in Docker for optimized environment

### API Key Issues

Verify keys are set:
```powershell
docker compose -f docker-compose.backtest.yml up backtest-diagnose
```

Look for "API_BASKETBALL_KEY: set" and "THE_ODDS_API_KEY: set" in output.
