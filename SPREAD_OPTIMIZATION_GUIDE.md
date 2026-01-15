# Spread Market Optimization Guide

## Overview

This guide explains how to run backtesting optimization for NBA spread markets (both Full Game and First Half). The optimization process tests different parameter combinations to identify the best settings for maximum profitability.

## What Gets Optimized

The optimization tests combinations of:

1. **Confidence Thresholds** (0.55, 0.60, 0.62, 0.65, 0.68, 0.70)
   - Minimum model confidence required to place a bet
   - Higher values = fewer bets but potentially more accurate

2. **Juice Values** (-105, -110, -115)
   - The odds/vig charged by sportsbooks
   - Lower (less negative) values = better profitability per win

3. **Markets** (FG Spread, 1H Spread)
   - Tested independently to find optimal parameters for each

## Quick Start

### Option 1: Python Script (Recommended)

Run the optimization with default parameters:

```bash
python scripts/run_spread_optimization.py
```

Quick test with fewer combinations:

```bash
python scripts/run_spread_optimization.py --quick
```

Custom configuration:

```bash
python scripts/run_spread_optimization.py \
  --markets fg_spread,1h_spread \
  --confidence 0.55,0.60,0.65,0.70 \
  --juice -105,-110
```

### Option 2: Batch Script (Windows)

```cmd
run_spread_optimization.bat
```

## Analyzing Results

After running the optimization, analyze the results:

```bash
python scripts/analyze_spread_optimization.py
```

This generates a comprehensive report showing:
- Top 10 configurations by ROI
- Top 5 configurations by accuracy
- Top 5 configurations by total profit
- Recommended configuration (balanced score)
- Parameter sensitivity analysis
- Final recommendations for production

## Output Files

Results are saved to: `data/backtest_results/spread_optimization/`

### Individual Result Files

Format: `{market}_conf{XX}_j{XXX}.json`

Examples:
- `fg_spread_conf55_j110.json` - FG spread with 55% confidence, -110 juice
- `1h_spread_conf62_j105.json` - 1H spread with 62% confidence, -105 juice

Each file contains:
- Number of bets
- Win rate / accuracy
- ROI (Return on Investment)
- Total profit in units
- Breakdown by season
- High-confidence subset metrics (60%+ confidence)

### Analysis Report

File: `data/backtest_results/spread_optimization_report.txt`

Contains:
- Ranked results by different metrics
- Parameter sensitivity analysis
- Final recommendations for production use

## Understanding the Metrics

### Key Metrics

1. **N Bets**: Total number of bets made
   - More bets = more statistical significance
   - Minimum 30-50 bets recommended for reliable metrics

2. **Accuracy**: Win rate percentage
   - Need >52.4% to be profitable at -110 juice
   - Need >51.2% to be profitable at -105 juice

3. **ROI**: Return on Investment percentage
   - Profit per unit wagered
   - Positive ROI = profitable strategy
   - Target: >2-5% ROI for sustainable betting

4. **Profit**: Total profit in units
   - Assumes 1 unit per bet
   - Shows absolute profitability

5. **Composite Score**: ROI × Accuracy × log(N_Bets)
   - Balances profitability, reliability, and sample size
   - Used to find "best balanced" configuration

### Confidence Tiers

Results are broken down by confidence levels:
- **55-60%**: Entry-level confidence bets
- **60-65%**: Medium confidence bets
- **65-70%**: High confidence bets
- **70%+**: Very high confidence bets

Higher tiers typically show:
- Fewer bets
- Higher accuracy
- Better ROI

## Typical Results

Based on NBA spread betting patterns, expect:

### FG Spread
- **Optimal confidence**: 0.60-0.65
- **Expected accuracy**: 53-56%
- **Expected ROI**: 2-6%
- **Bet volume**: 50-150 bets per season

### 1H Spread
- **Optimal confidence**: 0.65-0.70
- **Expected accuracy**: 54-57%
- **Expected ROI**: 3-7%
- **Bet volume**: 30-100 bets per season

## Interpreting Recommendations

The analysis script provides recommendations based on:

1. **Highest ROI**: Maximum profitability (may have low volume)
2. **Highest Accuracy**: Most reliable predictions
3. **Highest Profit**: Maximum absolute return
4. **Best Balanced**: Optimal trade-off (recommended for production)

### Production Deployment

Use the "Best Balanced" recommendation from the report:

1. Update `src/config.py` with recommended thresholds:
   ```python
   # Example for FG Spread
   spread_min_confidence: float = 0.62  # From optimization
   spread_min_edge: float = 2.0         # From optimization
   ```

2. Update environment variables:
   ```bash
   FILTER_SPREAD_MIN_CONFIDENCE=0.62
   FILTER_SPREAD_MIN_EDGE=2.0
   FILTER_1H_SPREAD_MIN_CONFIDENCE=0.68
   FILTER_1H_SPREAD_MIN_EDGE=1.5
   ```

## Advanced Usage

### Custom Date Ranges

Test specific seasons or date ranges:

```bash
python scripts/backtest_production.py \
  --markets fg_spread \
  --spread-juice -110 \
  --total-juice -110 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Single Configuration Test

Test a specific parameter combination:

```bash
python scripts/backtest_production.py \
  --markets fg_spread \
  --spread-juice -105 \
  --total-juice -110 \
  --output-json results_test.json
```

### Accuracy-Only Mode

Skip ROI calculations (useful for initial testing):

```bash
python scripts/backtest_production.py \
  --markets spreads \
  --no-pricing
```

## Files Created by This Optimization

### Scripts
1. `scripts/run_spread_optimization.py` - Main optimization runner
2. `scripts/analyze_spread_optimization.py` - Results analysis
3. `scripts/optimize_spread_parameters.py` - Alternative implementation
4. `run_spread_optimization.bat` - Windows batch runner

### Output
1. `data/backtest_results/spread_optimization/*.json` - Individual results
2. `data/backtest_results/spread_optimization/*.log` - Execution logs
3. `data/backtest_results/spread_optimization_report.txt` - Analysis report

## Troubleshooting

### No Bets Generated

If a configuration produces 0 bets:
- Confidence threshold too high
- Edge threshold too high
- Insufficient data meeting criteria

Try lowering confidence or edge thresholds.

### Low Sample Size

If results show <30 bets:
- Consider using lower confidence threshold
- Results may not be statistically significant
- Use longer date range for backtesting

### Poor Performance

If ROI is negative:
- Model may need retraining
- Check data quality
- Verify juice values match actual sportsbook odds
- Consider higher confidence thresholds

## Independent Operation

This optimization is designed to work independently:

✅ **Safe for parallel execution**
- Uses separate output files
- No shared state with other optimizations
- Can run simultaneously with total/moneyline optimization

✅ **No file conflicts**
- Reads from: `data/processed/master_training_data.csv` (read-only)
- Reads from: `models/production/` (read-only)
- Writes to: `data/backtest_results/spread_optimization/` (unique directory)

## Next Steps

1. Run the optimization: `python scripts/run_spread_optimization.py`
2. Analyze results: `python scripts/analyze_spread_optimization.py`
3. Review the report: `data/backtest_results/spread_optimization_report.txt`
4. Implement recommended parameters in production configuration
5. Monitor live performance against backtest expectations

## Support

For questions or issues:
1. Check the execution logs in `data/backtest_results/spread_optimization/*.log`
2. Verify data file exists: `data/processed/master_training_data.csv`
3. Verify models exist: `models/production/fg_spread_model.joblib` and `1h_spread_model.joblib`
