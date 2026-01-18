# Dynamic Model Retraining Plan

**Deployed Version**: NBA_v33.0.23.0  
**Deployment Date**: January 18, 2026  
**Markets Deployed**: 4 (1h_spread, 1h_total, fg_spread, fg_total)

## Current Model Performance

| Market | Test Accuracy | Baseline ROI | High-Conf Accuracy | High-Conf ROI | Status |
|--------|--------------|--------------|-------------------|---------------|--------|
| FG Spread | 66.2% | +26.5% | 76.1% | +45.3% | ✅ Excellent |
| FG Total | 59.3% | +13.2% | 67.9% | +29.6% | ✅ Solid |
| 1H Spread | 52.1% | -0.5% | 68.6% | +30.9% | ✅ Viable (high-conf) |
| 1H Total | 50.3% | -4.0% | 38.5% | -26.6% | ⚠️ Experimental |

## Retraining Strategy

### Frequency: Weekly During Season

**Every Monday (after weekend games):**
1. Fetch new game results from past week
2. Append to training dataset
3. Retrain all 4 models
4. Compare new vs old performance on hold-out set
5. Deploy if improvement detected OR every 2 weeks regardless

### Trigger Conditions for Immediate Retraining

**Retrain immediately if:**
- 1H Total accuracy improves above 52% on rolling 50-game window
- Any market drops below 50% accuracy on rolling 100-game window
- New data sources become available (better injury data, lineup data)
- Major NBA rule changes (e.g., shot clock adjustments)

### Training Commands

```bash
# 1. Update training data (manual or automated)
python scripts/generate_training_data.py

# 2. Train all 4 markets
python scripts/train_models.py --market all --ensemble

# 3. Validate performance
python scripts/validate_models.py --compare-versions

# 4. If validated, deploy
VERSION=$(cat VERSION)
docker build -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION -f Dockerfile.combined .
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION
```

## Data Collection for 1H Total Improvement

### Track These Metrics Weekly

**1H Total Performance Indicators:**
- Rolling 50-game accuracy (target: break above 52%)
- Confidence calibration (60%+ conf should = 60%+ accuracy)
- Brier score trend (lower = better calibration)
- High-confidence bet count (need at least 20 bets/week to validate)

**Dataset Growth:**
- Training data expands every week (more 1H samples)
- By mid-season: +800 1H games = 2x current dataset
- By end-season: +1,640 1H games = 3x current dataset

### Expected 1H Total Trajectory

| Date | Games Added | Total 1H Games | Expected Accuracy | Notes |
|------|-------------|----------------|------------------|-------|
| Jan 18, 2026 | 0 | 2,396 | 50.3% | Baseline (too noisy) |
| Feb 1, 2026 | +150 | 2,546 | 50-51% | Minimal improvement expected |
| Mar 1, 2026 | +450 | 2,846 | 51-52% | Signal may start emerging |
| Apr 1, 2026 | +750 | 3,146 | 52-53% | Target threshold (viable) |
| End Season | +1,640 | 4,036 | 53-54% | Full season signal |

**Key Insight**: 1H Total may become viable by **April 2026** as dataset grows to 3,000+ games.

## Automated Retraining Pipeline (Future)

### Phase 1: Manual (Current)
- Human reviews game results weekly
- Manually runs training scripts
- Validates performance before deployment

### Phase 2: Semi-Automated (Q2 2026)
- GitHub Action triggers on new data push
- Automatic training + validation
- Human approval required for deployment

### Phase 3: Fully Automated (Q3 2026)
- Scheduled nightly data ingestion
- Auto-retraining when threshold reached (e.g., +100 new games)
- Auto-deployment if validation passes (≥2% accuracy improvement OR no degradation)
- Slack/Teams notification of model updates

## Monitoring & Alerts

### Set up alerts for:
1. **Accuracy drops below baseline** (any market < 50%)
2. **1H Total improvement detected** (accuracy > 52% on 50-game rolling)
3. **High-confidence filter broken** (conf ≥60% but accuracy < 55%)
4. **Data pipeline failures** (no new games ingested for 7+ days)

### Dashboard Metrics (Weekly Review)
- Per-market accuracy trend (last 4 weeks)
- ROI trend (last 4 weeks)
- High-confidence bet volume (is filter too aggressive?)
- Model version changelog

## Rollback Plan

**If deployed model underperforms:**
1. Revert to previous Docker image:
   ```bash
   az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
     --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.22.0
   ```
2. Investigate degradation cause
3. Fix training data or feature engineering
4. Retrain and redeploy

**Version History:**
- Keep last 5 Docker images in ACR
- Keep last 10 model checkpoints in `models/production/archive/`
- Keep full training logs in `data/backtest_results/`

## Next Steps

1. **Week 1 (Jan 18-24)**: Monitor all 4 markets in production, collect real-time accuracy
2. **Week 2 (Jan 25-31)**: First manual retrain with +150 new games
3. **Week 4 (Feb 8-14)**: Evaluate 1H Total improvement trend
4. **Week 8 (Mar 7-13)**: Decision point: Keep or disable 1H Total based on accuracy
5. **End of Season (Apr 2026)**: Full season retrain, archive models, document learnings

## Resources

- Training data: `data/processed/training_data.csv`
- Model registry: `data/processed/models/model_registry.json`
- Backtest results: `data/backtest_results/`
- Deployment logs: Azure Container App logs
- Performance tracking: `MONITORING_SETUP_COMPLETE.md`

---

**Status**: ✅ All 4 markets deployed and ready for dynamic retraining  
**Next Retrain**: Week of January 25, 2026  
**Critical Watch**: 1H Total accuracy trend (goal: break 52% by April)
