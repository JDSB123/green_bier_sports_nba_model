# Next Steps: What To Do Now

**Status:** ✅ Production Ready (Docker-First)

---

## Current Status

### ✅ What's Working

- **Docker Stack:** All services containerized and running
- **Prediction API:** 6 backtested markets with strong ROI
- **Backtest Pipeline:** Containerized data + training + validation
- **Analysis Script:** Docker-only analysis with summary tables

### 📊 Performance

| Market | Accuracy | ROI |
|--------|----------|-----|
| FG Spread | 60.6% | +15.7% |
| FG Total | 59.2% | +13.1% |
| FG Moneyline | 65.5% | +25.1% |
| 1H Spread | 55.9% | +8.2% |
| 1H Total | 58.1% | +11.4% |
| 1H Moneyline | 63.0% | +19.8% |

---

## What To Do Now

### Option 1: Daily Predictions (Recommended)

**Goal:** Start using the system for daily predictions

**Steps:**

```powershell
# 1. Start the stack
docker compose up -d

# 2. Check health
curl http://localhost:8090/health

# 3. Get today's analysis
python scripts/analyze_slate_docker.py --date today

# 4. View predictions
curl http://localhost:8090/slate/today

# Results saved to:
# - data/processed/slate_analysis_YYYYMMDD.txt
# - data/processed/slate_analysis_YYYYMMDD.json
```

---

### Option 2: Run Full Backtest

**Goal:** Validate model performance with fresh data

**Steps:**

```powershell
# Run full backtest pipeline
docker compose -f docker-compose.backtest.yml up backtest-full

# View results
cat data/results/backtest_report_*.md
```

---

### Option 3: Extend to More Markets

**Goal:** Add Q1 markets or other bet types

**Steps:**

1. **Update training data with Q1 outcomes:**
   ```powershell
   docker compose -f docker-compose.backtest.yml run --rm backtest-shell
   # Inside container:
   python scripts/generate_q1_training_data.py
   ```

2. **Run Q1 backtest:**
   ```powershell
   docker compose -f docker-compose.backtest.yml up backtest-full
   # With MARKETS=q1_spread,q1_total,q1_moneyline
   ```

---

## Recommended Daily Workflow

### Morning (Before Games)

```powershell
# 1. Ensure stack is running
docker compose up -d

# 2. Get full analysis
python scripts/analyze_slate_docker.py --date today

# 3. Review picks in output
# Look for high fire ratings (🔥🔥🔥🔥 or 🔥🔥🔥🔥🔥)
```

### After Games

```powershell
# Results are tracked automatically via API
# Or check data/processed/ for prediction history
```

---

## Troubleshooting

### API Returns "Engine Not Loaded"

Models may be missing. Run backtest to regenerate:

```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

### Container Not Starting

```powershell
# Check logs
docker compose logs strict-api

# Common issues:
# - Missing .env file
# - Port 8090 already in use
# - Missing API keys
```

### No Games Found

```powershell
# Check if date is valid
curl "http://localhost:8090/slate/2025-12-18"

# Verify odds API is returning data
docker compose logs strict-api
```

---

## Summary

| Task | Command |
|------|---------|
| Start stack | `docker compose up -d` |
| Check health | `curl http://localhost:8090/health` |
| Get predictions | `python scripts/analyze_slate_docker.py --date today` |
| Run backtest | `docker compose -f docker-compose.backtest.yml up backtest-full` |
| Stop stack | `docker compose down` |

**Everything runs through Docker.** No local Python execution needed.
