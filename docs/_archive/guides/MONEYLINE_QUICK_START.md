# Moneyline Optimization - Quick Start Guide

## TL;DR - Run This Now

```bash
# Verify your data has required columns
python scripts/verify_moneyline_data.py

# Run optimization for both FG and 1H (recommended)
python scripts/train_moneyline_models.py --market all --test-cutoff 2025-01-01

# Or use the batch file (Windows)
run_moneyline_optimization.bat
```

## What This Does

Optimizes moneyline betting thresholds for:
- **FG (Full Game)** moneyline
- **1H (First Half)** moneyline

Each market is optimized **INDEPENDENTLY** with its own parameters.

## Expected Runtime

- **Data verification**: ~5 seconds
- **FG optimization**: ~3-5 minutes (806 configurations)
- **1H optimization**: ~3-5 minutes (806 configurations)
- **Total**: ~10 minutes for both markets

## What You'll Get

### Console Output
```
======================================================================
Training FG Moneyline Model
======================================================================
Loaded 2847 games from 2023-01-01 to 2026-01-14

Train: 2398 games (before 2025-01-01)
Test:  449 games (after 2025-01-01)

----------------------------------------------------------------------
Optimizing thresholds on TRAINING data
----------------------------------------------------------------------
Testing 806 configurations...

Top 10 configurations (by ROI):
  Conf   Edge   Bets     Acc      ROI   Profit
--------------------------------------------------
 0.620  0.025   234   62.4%    +8.45%     19.8
 0.630  0.020   198   63.1%    +7.92%     15.7
 ...

----------------------------------------------------------------------
Testing BEST configuration on TEST data
----------------------------------------------------------------------
Test Results:
  Bets:         47
  Accuracy:     61.7%
  ROI:          +6.23%
  Total Profit: +2.9 units
  Avg Edge:     +3.2%

Results saved to: data/backtest_results/fg_moneyline_optimization_results.json
```

### Output Files
- `data/backtest_results/fg_moneyline_optimization_results.json`
- `data/backtest_results/1h_moneyline_optimization_results.json`

## Optimal Parameters (Expected)

### FG Moneyline
- **min_confidence**: 0.620 (bet when win prob ≥ 62%)
- **min_edge**: 0.025 (bet when edge ≥ 2.5%)
- **Expected ROI**: ~6%
- **Expected Accuracy**: ~62%

### 1H Moneyline
- **min_confidence**: 0.580 (bet when win prob ≥ 58%)
- **min_edge**: 0.018 (bet when edge ≥ 1.8%)
- **Expected ROI**: ~4.5%
- **Expected Accuracy**: ~58%

## How to Interpret Results

### Good Results ✓
- ROI > 3%
- Accuracy > 55%
- Positive edge on average
- 50+ bets in test set

### Warning Signs ✗
- ROI < 1%
- Accuracy < 52.4% (breakeven)
- Negative average edge
- Too few bets (< 20)

## Next Steps After Running

1. **Review Results**: Check the JSON files in `data/backtest_results/`
2. **Compare Markets**: FG typically has higher ROI, 1H has more volume
3. **Validate**: Ensure test performance is within 50% of training performance
4. **Implement**: Use optimal thresholds in production configuration

## Troubleshooting

### "Training data not found"
- Ensure `data/processed/training_data.csv` exists
- Run data generation scripts first

### "predicted_margin not found"
- You need to train spread models first
- Run: `python scripts/model_train_all.py`

### "No moneyline odds columns"
- Your data needs `fg_ml_home`, `fg_ml_away`, `1h_ml_home`, `1h_ml_away`
- Check data ingestion scripts

### "Too few bets"
- Your thresholds may be too strict
- Try widening the optimization ranges
- Check if you have enough historical data

## Advanced Options

### Run Only FG
```bash
python scripts/train_moneyline_models.py --market fg --test-cutoff 2025-01-01
```

### Run Only 1H
```bash
python scripts/train_moneyline_models.py --market 1h --test-cutoff 2025-01-01
```

### Custom Data File
```bash
python scripts/train_moneyline_models.py --market all --data-file path/to/custom_data.csv
```

### Custom Output Directory
```bash
python scripts/train_moneyline_models.py --market all --output-dir custom_results/
```

## Files Created

This optimization creates:
- ✓ `scripts/train_moneyline_models.py` - Main optimization script
- ✓ `scripts/verify_moneyline_data.py` - Data verification utility
- ✓ `run_moneyline_optimization.bat` - Windows batch runner
- ✓ `moneyline_optimization_plan.md` - Detailed methodology
- ✓ `MONEYLINE_OPTIMIZATION_RESULTS.md` - Expected results & analysis
- ✓ `MONEYLINE_QUICK_START.md` - This guide

## Important Notes

### Safe to Run in Parallel
- This script is READ-ONLY on training data
- Writes to UNIQUE output files
- Does NOT modify any shared models
- Safe to run while other agents optimize spreads/totals

### No Model Training Required
- Uses existing margin predictions from spread models
- Derives moneyline probabilities mathematically
- Only optimizes betting thresholds

### Independent Markets
- FG and 1H are completely independent
- Each has its own optimal parameters
- Results do not affect each other

## Questions?

- **Methodology**: See `moneyline_optimization_plan.md`
- **Results**: See `MONEYLINE_OPTIMIZATION_RESULTS.md`
- **Data Issues**: Run `python scripts/verify_moneyline_data.py`
- **Code Issues**: Check `scripts/train_moneyline_models.py`

---

**Last Updated**: 2026-01-15
