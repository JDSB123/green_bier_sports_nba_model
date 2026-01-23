# Week 1 Monitoring Guide: NBA_v33.0.21.0

**Monitoring Period:** January 16-23, 2026
**Optimization:** FG Spread thresholds (0.62→0.55 conf, 2.0→0.0 edge)
**Goal:** Validate optimization before expanding to other markets

---

## 📊 Quick Start: Daily Monitoring

### Step 1: Save Today's Picks (Daily)

```bash
python scripts/predict_unified_save_daily_picks.py
```

This downloads and saves today's picks to `data/picks/picks_YYYY-MM-DD.json`.

### Step 2: View Daily Report (Daily)

```bash
python scripts/monitor_week1_performance.py --date 2026-01-16
```

Shows:
- Total picks and market breakdown
- FG Spread volume (target: 7-10/day)
- Average confidence and edge
- Status: ON TRACK / ACCEPTABLE / BELOW TARGET

### Step 3: Weekly Summary (After 7 Days)

```bash
python scripts/monitor_week1_performance.py --week-summary
```

Shows:
- Weekly totals across all markets
- FG Spread validation status
- Decision recommendation (deploy Option B or investigate)

---

## ✅ Success Criteria

### Volume Target
- **Goal:** 50-70 FG Spread picks per week
- **Daily:** 7-10 picks/day
- **Status Thresholds:**
  - ✅ ON TRACK: 7+ picks/day
  - ⚠️ ACCEPTABLE: 5-6 picks/day
  - ❌ BELOW TARGET: <5 picks/day

### Performance Target (Week 1)
- **ROI:** Above 20% (target: 27%)
- **Accuracy:** Above 63% (target: 65%)
- **No Catastrophic Losses:** Keep swings under 10 units

---

## 🚀 Next Steps After Week 1

### If Week 1 PASSES (50+ picks, good performance)

**Deploy Option B: FG Totals Optimization**

```bash
# 1. Validate what will change
python scripts/deploy_option_b.py --validate-only

# 2. Deploy to production
python scripts/deploy_option_b.py --deploy
```

**Expected Impact:**
- FG Total: 0.72→0.55 conf, 3.0→0.0 edge
- Expected ROI: +12.12%
- Expected Volume: ~2,721 bets/season

### If Week 1 is ACCEPTABLE (35-49 picks)

**Monitor 2-3 More Days**
- Continue daily monitoring
- Look for stabilization
- Deploy Option B if volume improves

### If Week 1 is BELOW TARGET (<35 picks)

**Investigate Issues**

1. **Check production thresholds:**
   ```bash
   az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.template.containers[0].env
   ```

2. **Verify API is using new thresholds:**
   ```bash
   curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health
   ```

3. **Review game filtering logic:**
   - Are games being filtered correctly?
   - Are predictions meeting confidence threshold?

4. **Consider adjustment options:**
   - Add slight edge filter (0.5-1.0 pts) for quality control
   - Use intermediate thresholds (0.58 conf, 1.0 edge)
   - Full rollback to v33.0.20.0

---

## 🔄 Rollback Options

If optimization is underperforming, three rollback options:

### Option 1: Full Rollback (Safest)
```bash
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.20.0 \
  --set-env-vars "FILTER_SPREAD_MIN_CONFIDENCE=0.62" \
                 "FILTER_SPREAD_MIN_EDGE=2.0"
```

### Option 2: Add Edge Filter (Moderate)
```bash
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --set-env-vars "FILTER_SPREAD_MIN_EDGE=1.5"
```

### Option 3: Intermediate Thresholds (Balanced)
```bash
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --set-env-vars "FILTER_SPREAD_MIN_CONFIDENCE=0.60" \
                 "FILTER_SPREAD_MIN_EDGE=1.0"
```

---

## 📁 File Locations

### Picks Data
- **Directory:** `data/picks/`
- **Format:** `picks_YYYY-MM-DD.json`
- **Source:** Production API downloads

### Monitoring Scripts
- **Save Picks:** `scripts/predict_unified_save_daily_picks.py`
- **Daily Report:** `scripts/monitor_week1_performance.py --date YYYY-MM-DD`
- **Weekly Summary:** `scripts/monitor_week1_performance.py --week-summary`
- **Deploy Option B:** `scripts/deploy_option_b.py`

### Documentation
- **Final Status:** `FINAL_STATUS_v33.0.21.0.md`
- **Deployment Summary:** `DEPLOYMENT_SUMMARY_v33.0.21.0.md`
- **Optimization Results:** `OPTIMIZATION_RESULTS_SUMMARY.md`

---

## 📈 Expected Timeline

### Week 1 (Jan 16-23): FG Spread Validation
- Daily monitoring of volume and quality
- Track actual vs expected performance
- Decision point: Deploy Option B or adjust

### Week 2-4 (Jan 23 - Feb 13): Option B Monitoring
- Deploy FG Totals optimization (if Week 1 successful)
- Monitor both FG Spread and FG Total
- Paper trade FG Moneyline

### Month 1 (Jan 16 - Feb 16): Full Validation
- Complete performance cycle
- Prepare Option C (Moneyline) deployment
- Plan quarterly reoptimization

---

## 🎯 Key Metrics to Track

### Volume Metrics
- Total picks per day
- FG Spread picks per day (CRITICAL)
- Market distribution

### Performance Metrics (Manual Tracking)
- Win/loss record for FG Spread
- Actual ROI vs expected (27%)
- Actual accuracy vs expected (65%)
- Closing line value

### Quality Metrics
- Average confidence of FG Spread picks
- Average edge of FG Spread picks
- Fire rating distribution

---

## 💡 Pro Tips

1. **Save picks daily** - Don't wait until end of week
2. **Track results manually** - Keep a simple spreadsheet of wins/losses
3. **Compare to baseline** - FG Spread was ~15% ROI before, should be ~27% now
4. **Watch for patterns** - Are we getting better closing line value?
5. **Be patient** - 7 days is minimum validation period

---

## 📞 Quick Reference Commands

```bash
# Save today's picks
python scripts/predict_unified_save_daily_picks.py

# View today's report
python scripts/monitor_week1_performance.py --date $(date +%Y-%m-%d)

# View weekly summary
python scripts/monitor_week1_performance.py --week-summary

# Check production status
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health

# Get today's picks from API
curl https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/slate/today

# Validate Option B deployment
python scripts/deploy_option_b.py --validate-only

# Deploy Option B (after Week 1 validation)
python scripts/deploy_option_b.py --deploy
```

---

**Remember:** The goal is to validate that FG Spread optimization works before expanding to other markets. Conservative, data-driven decisions are key to long-term success.

*Last Updated: 2026-01-16*
