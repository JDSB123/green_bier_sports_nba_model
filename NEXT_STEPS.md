# What's Next: Execution & Deployment Roadmap

**Current Status:** ✅ All optimization frameworks built and committed (v33.0.20.0)

---

## Immediate Next Steps (Now - Next 2 Hours)

### 1. Execute All 3 Optimizations in Parallel

Open 3 terminals and run:

**Terminal 1: Spreads**
```bash
cd c:\Users\JDSB\dev\green_bier_sport_ventures\nba_gbsv_local
python scripts/run_spread_optimization.py
```

**Terminal 2: Totals**
```bash
cd c:\Users\JDSB\dev\green_bier_sport_ventures\nba_gbsv_local
python scripts/optimize_totals_only.py
```

**Terminal 3: Moneylines**
```bash
cd c:\Users\JDSB\dev\green_bier_sport_ventures\nba_gbsv_local
python scripts/train_moneyline_models.py --market all --test-cutoff 2025-01-01
```

**Expected Runtime:** 30-45 minutes (all running in parallel)

---

### 2. Analyze Results

After optimizations complete:

```bash
# Analyze spreads
python scripts/analyze_spread_optimization.py

# Analyze totals
python scripts/analyze_totals_results.py

# Moneylines - review JSON files directly
cat data/backtest_results/fg_moneyline_optimization_results.json | jq '.top_10'
cat data/backtest_results/1h_moneyline_optimization_results.json | jq '.top_10'
```

---

### 3. Review Output Files

Check these locations:
- `data/backtest_results/spread_optimization/` - All spread results
- `data/backtest_results/totals_optimization_*.json` - Totals results
- `data/backtest_results/*_moneyline_optimization_results.json` - ML results
- `data/backtest_results/spread_optimization_report.txt` - Spreads summary
- `data/backtest_results/totals_optimization_summary.json` - Totals summary

---

## Short-Term Actions (Today - Tomorrow)

### 4. Validate Results Against Benchmarks

Compare actual results to expected performance:

| Market | Expected ROI | Expected Accuracy | Min Bets |
|--------|-------------|-------------------|----------|
| FG Spread | 2-6% | 53-56% | 50 |
| 1H Spread | 3-7% | 54-57% | 30 |
| FG Total | 2-5% | 54-58% | 50 |
| 1H Total | 1.5-4.5% | 53-57% | 50 |
| FG ML | 6.23% | 61.7% | 40 |
| 1H ML | 4.52% | 58.1% | 50 |

**Success Criteria:**
- ✅ ROI > 2% on test set
- ✅ Accuracy > 52.4% (breakeven at -110)
- ✅ Train/test ROI within 50% of each other
- ✅ Sufficient bet volume (30+ bets minimum)

---

### 5. Select Optimal Parameters

For each market, choose parameters that balance:
1. **ROI** (primary) - maximize profitability
2. **Volume** (secondary) - ensure sufficient bets for statistics
3. **Accuracy** (tertiary) - consistency matters

Document your choices in a new file:
```bash
# Create production config
cat > config/optimal_thresholds.json << 'EOF'
{
  "spreads": {
    "fg": {"min_confidence": 0.XX, "juice": -110},
    "1h": {"min_confidence": 0.XX, "juice": -110}
  },
  "totals": {
    "fg": {"min_confidence": 0.XX, "min_edge": X.X},
    "1h": {"min_confidence": 0.XX, "min_edge": X.X}
  },
  "moneylines": {
    "fg": {"min_confidence": 0.XX, "min_edge": 0.0XX},
    "1h": {"min_confidence": 0.XX, "min_edge": 0.0XX}
  }
}
EOF
```

---

### 6. Update Production Configuration

Edit `src/config.py` with optimal thresholds:

```python
class FilterThresholds:
    # SPREADS
    spread_min_confidence: float = 0.XX  # From optimization
    # ... update all thresholds

    # TOTALS
    total_min_confidence: float = 0.XX  # From optimization
    total_min_edge: float = X.X  # From optimization

    # ... etc for all markets
```

---

## Medium-Term Actions (This Week)

### 7. Validation Backtest on Recent Data

Test optimal parameters on most recent games (not used in optimization):

```bash
# Run validation backtest
python scripts/backtest_production.py \
  --data=data/processed/master_training_data.csv \
  --models-dir=models/production \
  --markets=all \
  --start-date=2025-12-01 \
  --output-json=data/backtest_results/validation_backtest.json
```

**Goal:** Confirm performance matches optimization results

---

### 8. Paper Trading (No Real Money)

Before deploying to production:

1. Set up paper trading environment
2. Track recommended bets for 50-100 games
3. Compare paper trading results to backtest expectations
4. Monitor for:
   - Systematic errors or biases
   - Line shopping opportunities
   - Closing line value
   - Market efficiency changes

**Duration:** 2-4 weeks (1 NBA week = ~50-60 games)

---

### 9. Set Up Monitoring & Alerts

Create monitoring dashboard:

```python
# Daily metrics to track
- Actual ROI vs expected ROI (by market)
- Win rate vs expected accuracy
- Bet volume vs expected volume
- Average edge vs sportsbook lines
- Closing line value (CLV)

# Alerts to configure
- ROI drops below 1% for 100+ bets
- Accuracy drops below 52% for 50+ bets
- Bet volume decreases >50% vs expected
- Model predictions deviate >20% from lines
```

---

## Long-Term Actions (Ongoing)

### 10. Production Deployment

When paper trading validates results:

1. Integrate optimal thresholds into production API
2. Set up automated bet recommendations
3. Implement bankroll management (Kelly Criterion)
4. Configure line shopping across multiple books
5. Enable real-money betting with conservative stakes

**Risk Management:**
- Start with 1% of bankroll per bet
- Gradually increase to optimal Kelly stakes (2-5%)
- Never exceed 5% of bankroll on single bet
- Diversify across markets (don't over-concentrate)

---

### 11. Performance Monitoring

**Daily:**
- Track actual vs predicted ROI
- Review bet recommendations and outcomes
- Check for line value vs closing lines

**Weekly:**
- Recalculate accuracy over trailing 7 days
- Compare to backtest expectations
- Check for seasonal patterns or shifts

**Monthly:**
- Full performance review vs optimization results
- Analyze edge degradation (market efficiency)
- Consider parameter adjustments if ROI drops >30%

---

### 12. Quarterly Recalibration

Every 3 months (or when performance degrades):

1. Re-run all 3 optimization scripts with updated data
2. Compare new optimal parameters to current settings
3. Analyze if market efficiency has changed
4. Update models if needed (retrain on recent data)
5. Adjust thresholds based on new optimization

**Triggers for Early Recalibration:**
- ROI drops below 2% for 200+ bets
- Win rate drops below 53% consistently
- Market conditions change significantly (rule changes, etc.)

---

## Optional Enhancements

### A. Advanced Features to Build

1. **Line Shopping Automation**
   - Integrate multiple sportsbook APIs
   - Auto-select best available odds
   - Expected ROI boost: +1-2%

2. **Closing Line Value (CLV) Tracking**
   - Compare bet placement odds to closing line
   - Positive CLV = sharp betting signal
   - Use for model validation

3. **Multi-Market Parlays**
   - Identify correlated edges across markets
   - Build +EV parlay combinations
   - Higher variance, higher potential ROI

4. **Live Betting Integration**
   - Extend models to in-game predictions
   - React to live odds movements
   - Requires real-time data feeds

5. **Ensemble Modeling**
   - Combine multiple model predictions
   - Weight by historical accuracy
   - Potential accuracy boost: +1-3%

---

### B. Data Pipeline Improvements

1. **Automate Training Data Updates**
   - Daily ingestion of new game results
   - Auto-retrain models weekly
   - Version control for datasets

2. **Feature Engineering**
   - Add travel/rest features (planned)
   - Incorporate referee assignments
   - Add player injury impact (advanced)

3. **Alternative Data Sources**
   - Weather data (outdoor events only - not NBA)
   - Social media sentiment
   - Sharp money tracking

---

## Success Metrics (90-Day Goals)

After 3 months of production betting:

| Metric | Target |
|--------|--------|
| **Portfolio ROI** | 3-5% |
| **Sharpe Ratio** | 0.3-0.5 |
| **Win Rate** | 54-60% |
| **Total Bets Placed** | 500-800 |
| **Closing Line Value** | Positive (>0%) |
| **Bankroll Growth** | 10-20% |

---

## Current File Status

✅ **Committed:**
- All optimization scripts (10 files)
- All batch runners (4 files)
- All documentation (8 files)
- Master summary (COMPLETE_OPTIMIZATION_SUMMARY.md)

✅ **Tagged:** v33.0.20.0

⏳ **Ready to Execute:**
- Run optimizations → Analyze results → Deploy parameters

---

## Key Resources

**Master Documentation:**
- [COMPLETE_OPTIMIZATION_SUMMARY.md](COMPLETE_OPTIMIZATION_SUMMARY.md) - Overview of all frameworks

**Quick Start Guides:**
- [SPREAD_OPTIMIZATION_GUIDE.md](SPREAD_OPTIMIZATION_GUIDE.md)
- [README_TOTALS_OPTIMIZATION.md](README_TOTALS_OPTIMIZATION.md)
- [MONEYLINE_QUICK_START.md](MONEYLINE_QUICK_START.md)

**Expected Results:**
- [TOTALS_OPTIMIZATION_RESULTS_TEMPLATE.md](TOTALS_OPTIMIZATION_RESULTS_TEMPLATE.md)
- [MONEYLINE_OPTIMIZATION_RESULTS.md](MONEYLINE_OPTIMIZATION_RESULTS.md)

---

## Questions to Consider

Before executing optimizations:

1. **Data Quality:**
   - ✅ Is master_training_data.csv current? (Yes - 3,195 games through 2025-26)
   - ✅ Are all leaky features removed? (Yes - v33.0.19.0 fix)
   - ✅ Are models trained on clean data? (Yes - retrained after leakage fix)

2. **Computational Resources:**
   - Do you have 30-45 minutes for parallel execution?
   - Sufficient disk space for results (~100MB)?
   - Python environment working?

3. **Production Readiness:**
   - How will you integrate optimal parameters?
   - Paper trading infrastructure ready?
   - Monitoring dashboard planned?

---

## Summary

**Status:** 🟢 Ready to execute optimizations

**Next Action:** Run all 3 optimization scripts in parallel (3 terminals)

**Time Required:** 30-45 minutes → then analyze results

**Expected Outcome:** Optimal parameters for all 6 markets, ready for production deployment

**Risk:** Low - frameworks are tested, documented, and use canonical data

---

*Generated: 2026-01-15*
*Version: v33.0.20.0*
*All systems ready for optimization execution* ✅
