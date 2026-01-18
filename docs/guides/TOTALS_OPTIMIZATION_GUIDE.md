# Totals Market Optimization Guide

## Overview

This document provides a comprehensive guide for optimizing betting parameters for **1H (First Half) Totals** and **FG (Full Game) Totals** markets independently.

## Objective

Find optimal confidence and edge thresholds that maximize ROI while maintaining acceptable bet volume and accuracy for totals betting.

## Current Thresholds (from config.py)

Based on the configuration in `src/config.py`:

### Full Game (FG) Totals
- **Min Confidence**: 0.72 (72%)
- **Min Edge**: 3.0 points

### First Half (1H) Totals
- **Min Confidence**: 0.66 (66%)
- **Min Edge**: 2.0 points

## Optimization Methodology

### 1. Parameter Grids

#### Confidence Grid
- **Range**: 0.55 to 0.79
- **Step**: 0.02
- **Values**: [0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73, 0.75, 0.77, 0.79]

#### Edge Grid
- **Range**: 0.0 to 6.0 points
- **Step**: 0.5 points
- **Values**: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

### 2. Total Combinations
- **Per Market**: 13 confidence × 13 edge = **169 combinations**
- **Both Markets**: 338 total combinations tested

### 3. Minimum Bet Threshold
- **Minimum**: 30 bets required for statistical significance
- Combinations with fewer bets are excluded from results

## How to Run the Optimization

### Option 1: Using the Batch File (Easiest)
```batch
# Simply double-click or run from command line
run_totals_optimization.bat
```

### Option 2: Using Python Directly
```bash
cd c:\Users\JDSB\dev\green_bier_sport_ventures\nba_gbsv_local
python scripts\optimize_totals_only.py
```

### Option 3: Using the Generic Optimization Script
```bash
# For FG totals only
python scripts/optimize_confidence_thresholds.py \
  --markets fg_total \
  --spread-juice -110 \
  --total-juice -110 \
  --confidence-min 0.55 \
  --confidence-max 0.79 \
  --confidence-step 0.02 \
  --edge-min 0.0 \
  --edge-max 6.0 \
  --edge-step 0.5 \
  --min-bets 30 \
  --objective roi \
  --output-json data/backtest_results/fg_total_optimization.json

# For 1H totals only
python scripts/optimize_confidence_thresholds.py \
  --markets 1h_total \
  --spread-juice -110 \
  --total-juice -110 \
  --confidence-min 0.55 \
  --confidence-max 0.79 \
  --confidence-step 0.02 \
  --edge-min 0.0 \
  --edge-max 6.0 \
  --edge-step 0.5 \
  --min-bets 30 \
  --objective roi \
  --output-json data/backtest_results/1h_total_optimization.json
```

## Output and Results

### Results Location
All results are saved to: `data/backtest_results/totals_optimization_YYYYMMDD_HHMMSS.json`

### Output Format
```json
{
  "timestamp": "ISO timestamp",
  "config": {
    "data_path": "data/processed/training_data.csv",
    "markets": ["fg_total", "1h_total"],
    "spread_juice": -110,
    "total_juice": -110,
    "confidence_grid": [...],
    "edge_grid": [...]
  },
  "baseline": {
    "fg_total": {
      "n_bets": 1234,
      "accuracy": 0.567,
      "roi": -2.34,
      "profit": -28.92
    },
    "1h_total": {...}
  },
  "optimization_results": {
    "fg_total": [
      {
        "confidence_threshold": 0.75,
        "edge_threshold": 3.5,
        "n_bets": 156,
        "wins": 92,
        "accuracy": 0.590,
        "roi": 4.23,
        "profit": 6.60
      },
      ...
    ],
    "1h_total": [...]
  }
}
```

### Key Metrics

For each parameter combination, the following metrics are calculated:

1. **n_bets**: Total number of bets (must be >= 30)
2. **wins**: Number of winning bets
3. **accuracy**: Win rate (wins / n_bets)
4. **roi**: Return on investment as percentage
5. **profit**: Total profit in units (risking 1 unit per bet)

### Ranking Criteria

Results are ranked by:
1. **Primary**: ROI (higher is better)
2. **Secondary**: Accuracy (higher is better)
3. **Tertiary**: Number of bets (higher is better)

## Analysis Considerations

### Trade-offs

When analyzing results, consider these trade-offs:

1. **ROI vs Volume**
   - Higher thresholds → Better ROI but fewer bets
   - Lower thresholds → More bets but potentially lower ROI

2. **Confidence vs Edge**
   - High confidence filter → Selects most certain predictions
   - High edge filter → Selects predictions with largest model advantage
   - Optimal balance varies by market

3. **Statistical Significance**
   - More bets → More reliable metrics
   - Fewer bets → Higher variance in results
   - Minimum 30 bets required, but 50+ preferred for stability

### Recommended Analysis Steps

1. **Review Top 10 Combinations**: Look at the top 10 parameter combinations for each market
2. **Check Bet Volume**: Ensure optimal parameters provide adequate bet volume (50+ bets preferred)
3. **Validate Across Seasons**: Check if parameters perform consistently across different seasons
4. **Compare to Baseline**: Measure improvement over unfiltered performance
5. **Consider Practical Constraints**: Balance optimization with real-world betting requirements

## Expected Outcomes

### Typical Patterns

Based on typical sports betting optimization:

#### FG Totals
- **Expected Optimal Confidence**: 0.67 - 0.75
- **Expected Optimal Edge**: 2.5 - 4.0 points
- **Target ROI**: 2% - 5%
- **Target Accuracy**: 54% - 58%

#### 1H Totals
- **Expected Optimal Confidence**: 0.63 - 0.71
- **Expected Optimal Edge**: 1.5 - 3.5 points
- **Target ROI**: 1.5% - 4.5%
- **Target Accuracy**: 53% - 57%

### Success Criteria

Parameters are considered successful if they achieve:
- ✅ ROI > 2%
- ✅ Accuracy > 52.4% (break-even at -110 odds)
- ✅ Bet volume > 50 bets for statistical significance
- ✅ Consistent performance across seasons

## Implementation

### Updating Configuration

After identifying optimal parameters, update `src/config.py`:

```python
@dataclass(frozen=True)
class FilterThresholds:
    # FG Total thresholds
    total_min_confidence: float = field(
        default_factory=lambda: _env_float_required("FILTER_TOTAL_MIN_CONFIDENCE", 0.XX)  # Update
    )
    total_min_edge: float = field(
        default_factory=lambda: _env_float_required("FILTER_TOTAL_MIN_EDGE", X.X)  # Update
    )

    # 1H Total thresholds
    fh_total_min_confidence: float = field(
        default_factory=lambda: _env_float_required("FILTER_1H_TOTAL_MIN_CONFIDENCE", 0.XX)  # Update
    )
    fh_total_min_edge: float = field(
        default_factory=lambda: _env_float_required("FILTER_1H_TOTAL_MIN_EDGE", X.X)  # Update
    )
```

### Testing New Parameters

Before deploying to production:

1. **Validate on Recent Data**: Test on most recent games not in training set
2. **Monitor Performance**: Track actual betting results vs expected
3. **Adjust as Needed**: Refine parameters based on live performance

## Troubleshooting

### Common Issues

1. **No combinations meet minimum bets**: Reduce min_bets threshold or expand parameter grids
2. **All ROIs negative**: Model may need retraining or market conditions have changed
3. **High variance in results**: Increase minimum bet threshold for more stable metrics
4. **Script errors**: Check Python environment and ensure all dependencies installed

### Getting Help

- Check logs in console output
- Review JSON output file for detailed results
- Validate training data is up to date
- Ensure production models are properly trained

## Notes

- This optimization is for **TOTALS ONLY** - spreads and moneylines are handled by other agents
- Results are based on historical data and may not perfectly predict future performance
- Regular re-optimization recommended (monthly or quarterly) as market conditions change
- Always validate optimized parameters on out-of-sample data before production deployment
