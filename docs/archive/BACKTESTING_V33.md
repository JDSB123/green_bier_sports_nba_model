# NBA Backtesting Engine v33.0.15.0

## Overview

Enterprise-grade backtesting framework for NBA prediction models with 6 independent markets, statistical validation, and Monte Carlo simulation.

**Version:** 33.0.15.0  
**Status:** Production Ready

---

## Quick Start

```bash
# Run all markets, full backtest
python scripts/backtest_v33.py

# Quick validation mode (FG markets only, faster)
python scripts/backtest_v33.py --quick

# Specific markets
python scripts/backtest_v33.py --markets fg_spread,1h_total

# Date range
python scripts/backtest_v33.py --start 2024-01-01 --end 2025-06-01
```

---

## Architecture

### Module Structure

```
src/
├── backtesting/
│   ├── __init__.py           # Module exports
│   ├── engine.py             # Core BacktestEngine orchestrator
│   ├── data_loader.py        # Data loading and validation
│   ├── walk_forward.py       # Hybrid walk-forward validation
│   ├── performance_metrics.py # ROI, Sharpe, drawdown, Kelly
│   ├── statistical_tests.py  # Enterprise statistical validation
│   ├── monte_carlo.py        # Monte Carlo simulation
│   └── report_generator.py   # HTML/Markdown reports
│
├── markets/
│   ├── __init__.py
│   ├── base.py               # BaseMarket abstract class
│   ├── spread.py             # Spread market (FG, 1H)
│   ├── total.py              # Total market (FG, 1H)
│   └── moneyline.py          # Moneyline market (FG, 1H)
│
scripts/
└── backtest_v33.py           # Single CLI entry point
```

### 6 Independent Markets

| Market | Period | Type | Description |
|--------|--------|------|-------------|
| `fg_spread` | Full Game | Spread | Home team covers the spread |
| `fg_total` | Full Game | Total | Game goes over/under |
| `fg_moneyline` | Full Game | Moneyline | Home team wins outright |
| `1h_spread` | First Half | Spread | Home team covers 1H spread |
| `1h_total` | First Half | Total | 1H goes over/under |
| `1h_moneyline` | First Half | Moneyline | Home team leads at halftime |

---

## Walk-Forward Methodology

### Hybrid Approach

The engine uses an **expanding window with recency weighting**:

1. **Expanding Window**: All historical data up to the test point is used for training
2. **Recency Weighting**: More recent games receive higher sample weights
3. **Periodic Retraining**: Model is retrained every N games (configurable)

```python
# Configuration
WalkForwardConfig(
    min_train_games=500,      # Minimum training set size
    test_chunk_size=50,       # Games per test chunk
    recency_weight_halflife=100,  # Games for 50% weight decay
    retrain_frequency=50,     # Retrain every N games
)
```

### Recency Weighting Formula

```python
def get_sample_weights(train_dates, current_date, halflife):
    days_ago = (current_date - train_dates).dt.days
    halflife_days = halflife * 2.5  # ~2.5 days per game
    weights = 0.5 ** (days_ago / halflife_days)
    return weights / weights.sum()  # Normalize
```

---

## Statistical Validation

### Hypothesis Tests

1. **t-test vs Random**: H0: accuracy = 50%
2. **t-test vs Profitable**: H0: accuracy <= 52.38% (break-even at -110)
3. **t-test ROI vs Break-even**: H0: ROI <= -4.55%
4. **Binomial Exact Test**: Exact test for small samples
5. **Runs Test**: Tests independence of outcomes

### Risk-Adjusted Metrics

- **Sharpe Ratio**: (mean_return - rf) / std_return
- **Sortino Ratio**: Downside deviation only
- **Max Drawdown**: Maximum peak-to-trough loss
- **Kelly Criterion**: Optimal bet fraction

### Confidence Intervals

- Bootstrap CI (95%) for accuracy and ROI
- Non-parametric, 10,000 bootstrap samples

---

## Monte Carlo Simulation

### Features

- Flat betting and Kelly betting strategies
- Probability of ruin calculation
- Bankroll projection distributions
- Sample path visualization

### Output Metrics

```python
MonteCarloResults(
    n_simulations=10000,
    probability_of_ruin=0.02,
    probability_of_profit=0.85,
    mean_final_bankroll=1500.0,
    percentile_5=800.0,
    percentile_95=2200.0,
    ...
)
```

---

## CLI Reference

```bash
python scripts/backtest_v33.py [OPTIONS]

Options:
  --data PATH           Training data CSV (default: data/processed/training_data_complete_2023.csv)
  --start DATE          Start date (YYYY-MM-DD)
  --end DATE            End date (YYYY-MM-DD)
  --markets LIST        Markets to test (default: all)
  --min-train N         Minimum training games (default: 500)
  --test-chunk N        Games per test chunk (default: 50)
  --retrain-freq N      Retrain every N games (default: 50)
  --halflife N          Recency weight halflife (default: 100)
  --model-type TYPE     logistic or gradient_boosting (default: logistic)
  --no-calibration      Disable probability calibration
  --min-confidence N    Minimum confidence threshold (default: 0.0)
  --min-edge N          Minimum edge threshold (default: 0.0)
  --monte-carlo N       Number of simulations (default: 10000, 0 to disable)
  --output-dir PATH     Output directory (default: data/backtest_results)
  --version TAG         Version tag (default: 33.0.15.0)
  --quick               Quick validation mode
  -v, --verbose         Verbose output
  -q, --quiet           Quiet mode
```

---

## Output Files

Each backtest run creates a timestamped directory:

```
data/backtest_results/v33.0.15.0/YYYY-MM-DD_HHMMSS/
├── predictions.csv           # All predictions with outcomes
├── backtest_summary.json     # Aggregated metrics (JSON)
├── backtest_report.html      # Visual HTML report
└── backtest_report.md        # Markdown summary
```

### Predictions CSV Schema

```csv
date,home_team,away_team,market,period,line,model_prob,prediction,actual,correct,profit,confidence,edge,kelly_fraction
```

---

## Key Principles

1. **NO SILENT FALLBACKS**: Fails loudly on bad data
2. **NO ASSUMPTIONS**: Uses real lines only (no FG/2 approximation for 1H)
3. **STATISTICALLY SIGNIFICANT**: Enterprise-level validation
4. **REPRODUCIBLE**: Versioned outputs with full audit trail
5. **MODULAR**: Each component independently testable

### Strict No-Fallback Policy Details

The backtest engine enforces strict data integrity:

| Scenario | Action | NO Fallback |
|----------|--------|-------------|
| Missing 1H line | Exclude game from 1H markets | ❌ No using FG line / 2 |
| Missing moneyline odds | Exclude game from ML market | ❌ No assuming -110 |
| Missing features | Exclude row from training | ❌ No zero-filling |
| Missing labels | Exclude game from backtest | ❌ No imputation |
| Invalid date | Exclude row entirely | ❌ No date estimation |

**How this affects results:**
- FG Moneyline: 2309 bets (656 games excluded due to missing odds)
- 1H Markets: 2285 bets (781 games excluded due to missing 1H data)
- Feature NaN: ~774 rows filtered per market

---

## Data Requirements

### Required Columns

- `date` or `game_date`: Game date
- `home_team`, `away_team`: Team names
- `home_score`, `away_score`: Final scores
- `fg_spread_line`, `fg_total_line`: Full game lines
- `1h_spread_line`, `1h_total_line`: First half lines

### Pre-computed Labels (Optional)

- `fg_spread_covered`, `fg_total_over`, `fg_home_win`
- `1h_spread_covered`, `1h_total_over`, `1h_home_win`

### Rolling Features (Used in Model)

- `home_5g_score`, `away_5g_score`, etc. (5/10/20 game windows)
- `diff_5g_score`, `diff_10g_score`, etc. (differentials)
- `home_elo`, `away_elo`, `elo_diff`
- `home_rest`, `away_rest`, `home_b2b`, `away_b2b`

---

## Example Results

```
============================================================
BACKTEST SUMMARY (All 6 Markets)
============================================================

FG_SPREAD
  Bets: 2292
  Accuracy: 51.7%
  ROI: -1.2%
  Sharpe: -0.01
  Status: NOT SIGNIFICANT

FG_TOTAL
  Bets: 2292
  Accuracy: 53.7%
  ROI: +2.5%
  Sharpe: 0.03
  Status: SIGNIFICANT

FG_MONEYLINE
  Bets: 2309
  Accuracy: 67.2%
  ROI: +3.7% (using actual odds, not -110)
  Sharpe: 0.02
  Status: SIGNIFICANT

1H_SPREAD
  Bets: 2285
  Accuracy: 51.5%
  ROI: -1.7%
  Sharpe: -0.02
  Status: NOT SIGNIFICANT

1H_TOTAL
  Bets: 2285
  Accuracy: 52.3%
  ROI: -0.2%
  Sharpe: -0.00
  Status: SIGNIFICANT

1H_MONEYLINE
  Bets: 2285
  Accuracy: 61.3%
  ROI: +6.3% (using actual odds, not -110)
  Sharpe: 0.02
  Status: NOT SIGNIFICANT
```

### Important Notes on Odds

- **Spread/Total markets**: Use standard -110 odds (industry standard)
- **Moneyline markets**: Use **actual odds from data** (e.g., -170, +150)
  - Games without valid moneyline odds are **excluded** (no fallback)
  - This is why moneyline accuracy is high but ROI is modest
  - Favorites have lower payouts, underdogs have higher payouts

---

## Troubleshooting

### "Missing feature columns"

Some features in the training data may be named differently. The engine filters to available features automatically.

### "Insufficient data for backtest"

Need at least `min_train_games + test_chunk_size` games. Reduce `--min-train` or use `--quick` mode.

### "NaN in features/labels"

Rows with NaN values are automatically filtered. Check data quality report for details.

---

## Migration from Legacy Scripts

The following legacy scripts have been replaced:

| Old Script | Replacement |
|------------|-------------|
| `scripts/backtest.py` | `scripts/backtest_v33.py` |
| `scripts/backtest_production_model.py` | `scripts/backtest_v33.py` |
| `scripts/walkforward_backtest_theodds.py` | `scripts/backtest_v33.py` |
| `scripts/quick_backtest.py` | `scripts/backtest_v33.py --quick` |
| `scripts/analyze_backtest_results.py` | Reports generated automatically |

---

## Version History

- **v33.0.15.0**: Complete rebuild with enterprise features
  - 6 independent markets
  - Hybrid walk-forward validation
  - Enterprise statistical validation
  - Monte Carlo simulation
  - HTML/Markdown/JSON reports

---

*Generated by NBA Backtest Engine v33.0.15.0*
