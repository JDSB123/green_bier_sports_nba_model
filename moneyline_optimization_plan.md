# Moneyline Optimization Plan

## Overview
This document outlines the approach to independently optimize moneyline betting for 1H (first half) and FG (full game) markets using margin-derived probabilities.

## Background
The current system has trained models for:
- FG Spread
- FG Total
- 1H Spread
- 1H Total

**Missing**: Moneyline models for both FG and 1H markets.

## Approach

### Methodology
Since spread models predict the margin (home_score - away_score), we can derive moneyline probabilities using:

```
P(home wins) = P(margin > 0) = Φ(predicted_margin / σ)
```

Where:
- Φ is the standard normal CDF
- σ = 12.0 for full game (based on NBA historical margin distribution)
- σ = 7.2 for first half (60% of full game variance)

### Key Advantages
1. **No New Models Required**: Leverages existing margin predictions from spread models
2. **Mathematically Sound**: Uses normal distribution assumption for point differentials
3. **Independent Optimization**: Each market (FG, 1H) optimized separately with own thresholds

## Optimization Parameters

### Parameters to Optimize
1. **min_confidence**: Minimum win probability required to place bet (range: 0.50 to 0.75)
2. **min_edge**: Minimum edge vs implied odds required (range: 0.0 to 0.15)

### Optimization Grid
- **Confidence**: 0.50 to 0.75 in steps of 0.01 (26 values)
- **Edge**: 0.0 to 0.15 in steps of 0.005 (31 values)
- **Total Configurations**: 806 per market

### Evaluation Metrics
- **Primary**: ROI (Return on Investment)
- **Secondary**: Accuracy, Total Profit
- **Constraints**: Minimum 30-50 bets required for valid configuration

## Execution Steps

### 1. Run FG Moneyline Optimization
```bash
python scripts/train_moneyline_models.py --market fg --test-cutoff 2025-01-01
```

**Expected Output**:
- Top 10 configurations ranked by ROI
- Best configuration tested on holdout set
- Performance metrics (accuracy, ROI, profit)
- Results saved to: `data/backtest_results/fg_moneyline_optimization_results.json`

### 2. Run 1H Moneyline Optimization
```bash
python scripts/train_moneyline_models.py --market 1h --test-cutoff 2025-01-01
```

**Expected Output**:
- Top 10 configurations ranked by ROI
- Best configuration tested on holdout set
- Performance metrics (accuracy, ROI, profit)
- Results saved to: `data/backtest_results/1h_moneyline_optimization_results.json`

### 3. Run Both Markets Together
```bash
python scripts/train_moneyline_models.py --market all --test-cutoff 2025-01-01
```

## Expected Results Format

### Training Output Example
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

Optimizing FG moneyline thresholds...
Testing 806 configurations...
  Progress: 5/26 confidence levels tested
  Progress: 10/26 confidence levels tested
  ...

Top 10 configurations (by ROI):
  Conf   Edge   Bets     Acc      ROI   Profit
--------------------------------------------------
 0.620  0.025   234   62.4%    +8.45%     19.8
 0.630  0.020   198   63.1%    +7.92%     15.7
 0.610  0.030   256   61.7%    +7.51%     19.2
 ...

----------------------------------------------------------------------
Testing BEST configuration on TEST data
----------------------------------------------------------------------
Best config: min_confidence=0.620, min_edge=0.025

Test Results:
  Bets:         47
  Accuracy:     61.7%
  ROI:          +6.23%
  Total Profit: +2.9 units
  Avg Edge:     +3.2%

Results saved to: data/backtest_results/fg_moneyline_optimization_results.json
```

### JSON Results Format
```json
{
  "period": "fg",
  "margin_std": 12.0,
  "best_config": {
    "min_confidence": 0.620,
    "min_edge": 0.025,
    "n_bets": 234,
    "accuracy": 0.6244,
    "roi": 0.0845,
    "total_profit": 19.8,
    "avg_edge": 0.0312
  },
  "test_metrics": {
    "n_bets": 47,
    "accuracy": 0.6170,
    "roi": 0.0623,
    "total_profit": 2.9,
    "avg_edge": 0.0320
  },
  "top_10_train_configs": [...]
}
```

## Data Requirements

### Required Columns in training_data.csv
**For FG Moneyline**:
- `predicted_margin` - Model's predicted margin (from spread model)
- `fg_ml_home` - American odds for home team moneyline
- `fg_ml_away` - American odds for away team moneyline
- `home_score` - Actual home team score
- `away_score` - Actual away team score
- `date` or `game_date` - Game date for temporal splitting

**For 1H Moneyline**:
- `predicted_margin_1h` - Model's predicted 1H margin
- `1h_ml_home` - American odds for home team 1H moneyline
- `1h_ml_away` - American odds for away team 1H moneyline
- `home_1h` OR (`home_q1` + `home_q2`) - Actual 1H home score
- `away_1h` OR (`away_q1` + `away_q2`) - Actual 1H away score
- `date` or `game_date` - Game date

## Interpretation Guide

### What Makes a Good Configuration?

1. **High ROI (Primary Goal)**
   - Target: >5% ROI
   - Good: 3-5% ROI
   - Acceptable: 1-3% ROI
   - Poor: <1% ROI

2. **Sufficient Volume**
   - Minimum: 30-50 bets
   - Good: 100+ bets
   - Excellent: 200+ bets

3. **High Accuracy**
   - Required: >52.4% (breakeven at -110)
   - Good: >55%
   - Excellent: >58%

4. **Positive Edge**
   - Our probability > Implied probability from odds
   - Target: >2% average edge

### Trade-offs
- **Higher confidence threshold** → Fewer bets, higher accuracy, potentially higher ROI
- **Higher edge threshold** → Fewer bets, better value, potentially higher ROI
- **Lower thresholds** → More bets, more variance, lower average edge

## Validation Steps

After optimization, validate results by:

1. **Check Train/Test Consistency**
   - Test ROI should be within 50% of train ROI
   - If test ROI << train ROI, may be overfitting

2. **Verify Bet Volume**
   - Ensure test set has enough bets (>20) for meaningful evaluation
   - Check that bet distribution matches expected frequency

3. **Review Edge Distribution**
   - Bets should have positive average edge
   - Edge should align with implied probability gap

4. **Compare Across Markets**
   - FG typically has more liquidity/data than 1H
   - 1H may have different optimal thresholds due to higher variance

## Next Steps After Optimization

1. **Document Optimal Parameters**
   - Record best configurations for each market
   - Save to configuration file for production use

2. **Implement in Production**
   - Add moneyline prediction to prediction engine
   - Use optimal thresholds for bet filtering
   - Monitor real-world performance

3. **Continuous Monitoring**
   - Track live performance vs backtest
   - Recalibrate thresholds quarterly
   - Adjust margin_std if distribution changes

## Files Created

1. `scripts/train_moneyline_models.py` - Main optimization script
2. `moneyline_optimization_plan.md` - This documentation
3. `data/backtest_results/fg_moneyline_optimization_results.json` - FG results (after running)
4. `data/backtest_results/1h_moneyline_optimization_results.json` - 1H results (after running)

## Notes

- **Independence**: This script works INDEPENDENTLY and will not interfere with other agents optimizing spreads/totals
- **No File Conflicts**: Results are saved to unique filenames per market
- **Read-Only Data**: Only reads from training_data.csv, does not modify it
- **Safe to Run Parallel**: Can run simultaneously with spread/total optimization
