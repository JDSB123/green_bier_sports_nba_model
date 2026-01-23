# Monitoring Setup Complete - NBA_v33.0.21.0

**Date:** 2026-01-16
**Status:** ✅ READY FOR WEEK 1 VALIDATION
**Git Commit:** d0cdb9b

---

## 📊 What Was Set Up

### 1. Week 1 Monitoring Framework
Created automated tools to track FG Spread optimization performance:

- **[scripts/predict_unified_save_daily_picks.py](scripts/predict_unified_save_daily_picks.py)** - Downloads daily picks from production API
- **[scripts/monitor_week1_performance.py](scripts/monitor_week1_performance.py)** - Analyzes daily/weekly performance
- **[WEEK1_MONITORING_GUIDE.md](WEEK1_MONITORING_GUIDE.md)** - Complete monitoring instructions

### 2. Option B Deployment Script
Created automated deployment pipeline for FG Totals expansion:

- **[scripts/deploy_option_b.py](scripts/deploy_option_b.py)** - One-command deployment of FG Totals optimization
- Validates changes before deployment
- Handles Docker build, ACR push, Azure deployment, and Git versioning

---

## 🚀 Quick Start: Daily Workflow

### Every Day During Week 1 (Jan 16-23)

```bash
# Step 1: Save today's picks (run once per day)
python scripts/predict_unified_save_daily_picks.py

# Step 2: View daily report
python scripts/monitor_week1_performance.py --date 2026-01-16
```

### After 7 Days (Jan 23)

```bash
# View weekly summary
python scripts/monitor_week1_performance.py --week-summary

# If validation passes, deploy Option B
python scripts/deploy_option_b.py --validate-only  # Preview changes
python scripts/deploy_option_b.py --deploy         # Deploy to production
```

---

## 🎯 Success Criteria Tracking

### Volume Target
- **Goal:** 50-70 FG Spread picks per week
- **Daily Target:** 7-10 picks/day
- **Monitoring:** Automated in daily report

### Performance Target
- **ROI:** Above 20% (target: 27%)
- **Accuracy:** Above 63% (target: 65%)
- **Tracking:** Manual (record wins/losses)

### Decision Points
- ✅ **50+ picks/week** → Deploy Option B immediately
- ⚠️ **35-49 picks/week** → Monitor 2-3 more days
- ❌ **<35 picks/week** → Investigate issues

---

## 📁 Files Created

### Monitoring Tools
```
scripts/
├── predict_unified_save_daily_picks.py              # Download daily picks from API
├── monitor_week1_performance.py     # Performance analysis
└── deploy_option_b.py               # Option B deployment automation

WEEK1_MONITORING_GUIDE.md            # Comprehensive guide
MONITORING_SETUP_COMPLETE.md         # This file
```

### Data Directory Structure
```
data/
└── picks/
    ├── picks_2026-01-16.json       # Created when you run predict_unified_save_daily_picks.py
    ├── picks_2026-01-17.json       # One file per day
    └── ...
```

---

## 🔄 Next Steps After Week 1

### If Validation PASSES (50+ picks, good performance)

**Deploy Option B: FG Totals Optimization**

Expected Changes:
- FG Total: 0.72 → 0.55 conf, 3.0 → 0.0 edge
- Expected ROI: +12.12%
- Expected Volume: ~2,721 bets/season
- Version: NBA_v33.0.21.0 → NBA_v33.0.22.0

Command:
```bash
python scripts/deploy_option_b.py --deploy
```

### If Validation is MIXED (35-49 picks)

Continue monitoring 2-3 more days, then reassess.

### If Validation FAILS (<35 picks)

Investigate:
1. Check Azure environment variables are active
2. Verify API is using new thresholds
3. Review game filtering logic
4. Consider rollback options (see WEEK1_MONITORING_GUIDE.md)

---

## 🛠️ Tools Available

### Daily Monitoring
```bash
# Save today's picks
python scripts/predict_unified_save_daily_picks.py

# View daily report for specific date
python scripts/monitor_week1_performance.py --date 2026-01-16

# View weekly summary (after 7 days)
python scripts/monitor_week1_performance.py --week-summary
```

### Option B Deployment
```bash
# Preview what Option B will change
python scripts/deploy_option_b.py --validate-only

# Deploy Option B to production
python scripts/deploy_option_b.py --deploy
```

### Production Status Checks
```bash
# Check API health
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health

# Get today's picks
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today

# Check Azure environment variables
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.template.containers[0].env
```

---

## 📋 Monitoring Checklist

### Week 1 Daily Tasks
- [ ] Day 1 (Jan 16): Save picks, view report
- [ ] Day 2 (Jan 17): Save picks, view report
- [ ] Day 3 (Jan 18): Save picks, view report
- [ ] Day 4 (Jan 19): Save picks, view report
- [ ] Day 5 (Jan 20): Save picks, view report
- [ ] Day 6 (Jan 21): Save picks, view report
- [ ] Day 7 (Jan 22): Save picks, view report

### Week 1 Summary Tasks
- [ ] Jan 23: Run weekly summary
- [ ] Evaluate success criteria (volume, ROI, accuracy)
- [ ] Make decision: Deploy Option B / Continue monitoring / Investigate

---

## 📚 Documentation References

**Primary Monitoring Guide:**
- [WEEK1_MONITORING_GUIDE.md](WEEK1_MONITORING_GUIDE.md) - Complete instructions

**System Documentation:**
- [FINAL_STATUS_v33.0.21.0.md](FINAL_STATUS_v33.0.21.0.md) - Current deployment status
- [DEPLOYMENT_SUMMARY_v33.0.21.0.md](DEPLOYMENT_SUMMARY_v33.0.21.0.md) - Deployment details
- [OPTIMIZATION_RESULTS_SUMMARY.md](OPTIMIZATION_RESULTS_SUMMARY.md) - Full optimization analysis

**Configuration:**
- [src/config.py](src/config.py) - Filter thresholds
- [models/production/model_pack.json](models/production/model_pack.json) - Model metadata
- [VERSION](VERSION) - Current version (NBA_v33.0.21.0)

---

## ✅ Setup Status

- ✅ Monitoring scripts created
- ✅ Deployment automation created
- ✅ Documentation complete
- ✅ Git committed and pushed
- ✅ Directory structure ready
- ✅ Tools tested

**System Status:** 🟢 **READY FOR WEEK 1 MONITORING**

---

## 🎯 Remember

1. **Monitor daily** - Don't wait until end of week
2. **Track results manually** - Keep a spreadsheet of wins/losses
3. **Be patient** - 7 days minimum for validation
4. **Data-driven decisions** - Let the numbers guide next steps

**Goal:** Validate FG Spread optimization works before expanding to other markets.

---

*Setup completed: 2026-01-16*
*Next review: 2026-01-23 (after Week 1)*
*Git commit: d0cdb9b*
