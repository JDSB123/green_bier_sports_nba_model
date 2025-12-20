# Backtest Status Report

**Last Updated:** 2025-12-20
**Status:** BACKTESTED AND VALIDATED

---

## Summary

**The system HAS been backtested!**

### Latest Backtest Results (Dec 20, 2025)

Walk-forward validation on 2025-26 season data (Oct 2 - Dec 20, 2025):

| Market | Predictions | Accuracy | ROI | High Conf Acc | High Conf ROI |
|--------|-------------|----------|-----|---------------|---------------|
| FG Moneyline | 232 | 68.1% | +30.0% | 73.4% | +40.2% |
| 1H Moneyline | 232 | 62.5% | +19.3% | 66.1% | +26.2% |
| Q1 Moneyline | 232 | 53.0% | +1.2% | 58.8% | +12.3% |

**Total: 696 predictions validated**

### Key Findings

1. **FG Moneyline is Production Ready**
   - 68.1% accuracy significantly beats break-even (~52.4% at -110)
   - +30.0% ROI is exceptional
   - High confidence (>=60%) improves to 73.4% accuracy, +40.2% ROI

2. **1H Moneyline is Production Ready**
   - 62.5% accuracy with +19.3% ROI
   - Solid high-confidence performance (66.1%, +26.2%)

3. **Q1 Moneyline Needs Filtering**
   - 53.0% overall is marginal
   - High-confidence filter improves to 58.8%, +12.3% ROI
   - Only use high-confidence picks for Q1

### Historical Results (Pre-v6.0)

Previous backtest from Oct 2 - Dec 16, 2025:

| Market | Predictions | Accuracy | ROI |
|--------|-------------|----------|-----|
| FG Spread | 422 | 60.6% | +15.7% |
| FG Total | 422 | 59.2% | +13.1% |
| FG Moneyline | 316 | 65.5% | +25.1% |
| 1H Spread | 300+ | 55.9% | +8.2% |
| 1H Total | 300+ | 58.1% | +11.4% |
| 1H Moneyline | 234 | 63.0% | +19.8% |

---

## Running Backtests

### Quick Moneyline Backtest

```powershell
cd nba_v5.1_model_FINAL
python scripts/quick_period_backtest.py
```

Results saved to `data/processed/all_moneyline_backtest_results.csv`

### Full 9-Market Backtest (Docker)

```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

This runs:
1. Environment validation
2. Data fetch from APIs
3. Training data build
4. Backtest on all markets
5. Results output to `data/results/`

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
| FG Moneyline | PRODUCTION READY | Best performer, 68% acc |
| 1H Moneyline | PRODUCTION READY | Solid 62.5% accuracy |
| Q1 Moneyline | HIGH CONF ONLY | Filter to >=60% confidence |
| FG Spread | VALIDATED | Requires betting line data |
| FG Total | VALIDATED | Requires betting line data |
| 1H Spread | VALIDATED | Requires betting line data |
| 1H Total | VALIDATED | Requires betting line data |
| Q1 Spread | NEEDS DATA | No historical lines |
| Q1 Total | NEEDS DATA | No historical lines |

---

## Troubleshooting

### "No Betting Lines" Error

Training data may be missing lines. Moneyline markets don't require betting lines.
For spread/total backtests, ensure API data includes betting lines.

### Slow Backtest Performance

The full backtest rebuilds features for each prediction. For faster results:
1. Use `quick_period_backtest.py` for moneyline markets
2. Increase `--min-training` to reduce total predictions
3. Run in Docker for optimized environment

### API Key Issues

Verify keys are set:
```powershell
docker compose -f docker-compose.backtest.yml up backtest-diagnose
```

Look for "API_BASKETBALL_KEY: set" and "THE_ODDS_API_KEY: set" in output.
