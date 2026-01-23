# Daily Workflow - NBA Model v33.0.21.0

**Version:** NBA_v33.0.21.0 (Optimized FG Spread)
**Status:** Production, Week 1 Validation
**Date Range:** Jan 16-23, 2026

---

## 📊 Daily Routine (5 Minutes)

### Step 1: Post Picks to Teams (30 seconds)
```bash
post_to_teams_simple.bat
```

### Step 2: Compare Versions (1 minute)
```bash
python scripts/compare_thresholds_api.py --date today
```

**What this shows:**
- Baseline (v20) vs Current (v21) pick counts
- FG Spread volume change (should be +200-300%)
- Validation that optimization is working

**Expected Results:**
- FG Spread: 7-10 picks/day (up from 2-3)
- Total picks: 20-30/day (up from 15-20)
- Status: ON TRACK or ACCEPTABLE

### Step 3: Save Picks for Monitoring (30 seconds)
```bash
python scripts/predict_unified_save_daily_picks.py
```

### Step 4: View Daily Report (1 minute)
```bash
python scripts/monitor_week1_performance.py --date 2026-01-16
```

**What this shows:**
- Pick counts by market
- FG Spread metrics (confidence, edge)
- Tracking toward weekly targets

---

## 📅 Weekly Review (Jan 23)

### After 7 Days
```bash
python scripts/monitor_week1_performance.py --week-summary
```

**Decision Points:**
- ✅ **50+ FG Spread picks/week** → Deploy Option B (FG Totals)
- ⚠️ **35-49 picks/week** → Monitor 2-3 more days
- ❌ **<35 picks/week** → Investigate issues

### If Week 1 Passes
```bash
# Deploy FG Totals optimization
python scripts/deploy_option_b.py --validate-only  # Preview
python scripts/deploy_option_b.py --deploy         # Deploy
```

---

## 🎯 Success Metrics

### Volume (Primary)
- **FG Spread:** 50-70 picks/week (7-10/day)
- **Baseline:** Was ~15 picks/week (2-3/day)
- **Target Increase:** 200-300%

### Performance (Secondary - Manual Tracking)
- **ROI:** Above 20% (target: 27%)
- **Accuracy:** Above 63% (target: 65%)
- **No major losses:** Keep swings under 10 units

---

## 🛠️ Complete Command Reference

### Daily Commands
```bash
# Post to Teams
post_to_teams_simple.bat

# Compare versions
python scripts/compare_thresholds_api.py --date today

# Save picks
python scripts/predict_unified_save_daily_picks.py

# Daily report
python scripts/monitor_week1_performance.py --date 2026-01-16

# Weekly summary
python scripts/monitor_week1_performance.py --week-summary
```

### Deployment Commands
```bash
# Option B (after Week 1 validation)
python scripts/deploy_option_b.py --validate-only
python scripts/deploy_option_b.py --deploy
```

### API Commands
```bash
# Check health
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health

# Get picks
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today

# Get executive summary
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today/executive
```

---

## 📁 Where Files Are Saved

### Daily Picks
```
data/picks/picks_2026-01-16.json
data/picks/picks_2026-01-17.json
...
```

### Documentation
```
WEEK1_MONITORING_GUIDE.md      - Complete monitoring guide
DAILY_WORKFLOW.md              - This file (quick reference)
MONITORING_SETUP_COMPLETE.md   - Setup summary
TEAMS_WEBHOOK_SETUP.md         - Teams integration details
QUICK_START_TEAMS.md           - Teams quick start
```

---

## 🚨 Troubleshooting

### FG Spread Volume Low (<7/day)

**Check:**
1. Azure environment variables are set
2. API is running v33.0.21.0
3. Thresholds are active (0.55 conf, 0.0 edge)

**Verify:**
```bash
# Check Azure env vars
az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.template.containers[0].env

# Check API health
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health
```

### Teams Post Not Working

**Solution:**
```bash
# Use API endpoint directly
curl -X POST "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/teams/outgoing" \
  -H "Content-Type: application/json" \
  -d '{"text":"picks"}'
```

### Comparison Script Shows No Change

**Possible causes:**
1. API not using new thresholds
2. No games today
3. Predictions not meeting thresholds

**Check:**
Run with verbose output to see raw predictions

---

## ✅ Daily Checklist

**Every Morning:**
- [ ] Post picks to Teams (`post_to_teams_simple.bat`)
- [ ] Run version comparison (`compare_thresholds_api.py`)
- [ ] Check FG Spread volume (should be 7-10)
- [ ] Save picks for tracking (`predict_unified_save_daily_picks.py`)
- [ ] View daily report (`monitor_week1_performance.py`)

**End of Week (Jan 23):**
- [ ] Run weekly summary
- [ ] Calculate actual ROI (manual)
- [ ] Decide on Option B deployment
- [ ] Document Week 1 results

---

## 🎯 Key Insight

**The version comparison tool is your daily validator.**

It shows you immediately whether the optimization is working by comparing what v20 would have produced vs what v21 actually produces. The key metric is **FG Spread volume** - if this isn't increasing 200-300%, something is wrong.

**Expected Pattern:**
- Baseline (v20): 1-3 FG Spread picks/day
- Current (v21): 7-10 FG Spread picks/day
- Increase: +200-300%

If you're not seeing this increase, investigate immediately - don't wait until end of week.

---

*Last Updated: 2026-01-16*
*Production Version: NBA_v33.0.21.0*
*Next Review: 2026-01-23*
