# NBA Predictions Quick Reference Card

## 🚀 Quick Commands

### Run Tonight's Predictions (Easiest)
```bash
./run_tonight_predictions.sh
```

### Run Tomorrow's Predictions
```bash
./run_tonight_predictions.sh tomorrow
```

### Filter by Team
```bash
./run_tonight_predictions.sh today "Lakers"
```

### Python Direct
```bash
python scripts/run_slate.py
python scripts/run_slate.py --date tomorrow
python scripts/run_slate.py --matchup "Celtics"
```

---

## 📋 Prerequisites Checklist

- [ ] Docker Desktop is running
- [ ] API keys configured (see below)
- [ ] Python 3.x installed

### API Keys Setup (One-time)
```bash
# Option 1: Docker Secrets (Recommended)
mkdir -p secrets
echo 'your_key' > secrets/THE_ODDS_API_KEY
echo 'your_key' > secrets/API_BASKETBALL_KEY

# Option 2: Environment File
cp .env.example .env
# Edit .env with your keys
```

---

## 🎯 What You Get

### Console Output
- **Summary table** - All picks at a glance
- **Detailed rationale** - Why each pick was made
- **Fire ratings** - 🔥 to 🔥🔥🔥🔥🔥 (strength indicator)
- **Edge values** - Expected value in points
- **Confidence** - Model certainty percentage

### Files Generated
```
data/processed/slate_output_20251231_183045.txt    # Text report
data/processed/slate_output_20251231_183045.html   # HTML report (styled)
archive/slate_outputs/                             # Archived copies
```

---

## 🔥 Fire Ratings Guide

| Rating | Meaning | Combined Score |
|--------|---------|----------------|
| 🔥🔥🔥🔥🔥 | Elite pick | 85%+ |
| 🔥🔥🔥🔥 | Strong pick | 70-85% |
| 🔥🔥🔥 | Good pick | 60-70% |
| 🔥🔥 | Moderate pick | 52-60% |
| 🔥 | Marginal pick | <52% |

**Combined Score** = (Confidence × 60%) + (Edge Normalized × 40%)

---

## 🎮 GitHub Actions (No Local Setup)

1. Go to repository **Actions** tab
2. Select **"Run NBA Predictions"**
3. Click **"Run workflow"**
4. Choose date: **today** or **tomorrow**
5. *(Optional)* Add matchup filter
6. Click **"Run workflow"** button
7. Wait ~1-2 minutes
8. Download **artifact** with predictions

---

## ⚙️ Container Management

### Check Status
```bash
docker ps --filter "name=nba-v33"
```

### View Logs
```bash
docker compose logs -f nba-v33
```

### Restart Container
```bash
docker compose restart
```

### Stop Container
```bash
docker compose down
```

### Start Container
```bash
docker compose up -d
```

---

## 🩺 Health Checks

### API Health
```bash
curl http://localhost:8090/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "engine_loaded": true,
  "models_loaded": 4,
  "api_keys_configured": true
}
```

### Docker Running
```bash
docker info
```

### Python Available
```bash
python3 --version
```

---

## 🐛 Common Issues

### "Docker is not running"
**Fix:** Start Docker Desktop, then retry

### "API did not become ready"
```bash
docker compose logs nba-v33
docker compose restart
```

### "No API keys found"
**Fix:** Configure secrets or .env (see Prerequisites)

### "No games found"
**Normal if:**
- Too early in day (odds not posted yet)
- No games scheduled
- Off-season

**Try:** `./run_tonight_predictions.sh tomorrow`

### Models not loaded
```bash
# Verify models exist
ls -lh models/production/

# If missing, train models
python scripts/train_models.py
```

---

## 📊 Output Format Examples

### Spread Pick
```
FG Spread: Cleveland Cavaliers -7.5 (-110)
   Model predicts: CLE wins by 9.2 pts
   Market line: -7.5
   Edge: +1.7 pts of value
   Confidence: 62% | 🔥🔥🔥
   EV: +3.4% | Kelly: 0.03
```

### Total Pick
```
FG Total: OVER 223.5 (-110)
   Model predicts: 227.3 total points
   Market line: 223.5
   Edge: 3.8 pts of value
   Confidence: 68% | 🔥🔥🔥🔥
   EV: +5.2% | Kelly: 0.05
```

---

## 🔗 Useful Links

- **Full Documentation:** `docs/RUNNING_PREDICTIONS.md`
- **Troubleshooting:** `docs/DOCKER_TROUBLESHOOTING.md`
- **Architecture:** `docs/STACK_FLOW_AND_VERIFICATION.md`
- **Main README:** `README.md`

---

## 📞 Quick Commands Reference

| Task | Command |
|------|---------|
| **Run predictions** | `./run_tonight_predictions.sh` |
| **Tomorrow's slate** | `./run_tonight_predictions.sh tomorrow` |
| **Filter team** | `./run_tonight_predictions.sh today "Lakers"` |
| **Check health** | `curl http://localhost:8090/health` |
| **View logs** | `docker compose logs -f nba-v33` |
| **Restart API** | `docker compose restart` |
| **Train models** | `python scripts/train_models.py` |
| **Run backtest** | `docker compose -f docker-compose.backtest.yml up backtest-full` |
| **Validate setup** | `python scripts/validate_production_readiness.py` |

---

## ⏱️ Typical Timing

- **Container startup:** 15-30 seconds (first time)
- **Health check:** 5-10 seconds
- **Predictions fetch:** 10-30 seconds per game
- **Full slate:** 1-2 minutes total

---

## 💡 Pro Tips

1. **Run early:** Fetch predictions 1-2 hours before games
2. **Check often:** Lines move, refresh for latest odds
3. **Archive enabled:** All outputs auto-saved to `archive/`
4. **HTML reports:** Open in browser for better viewing
5. **Filter wisely:** Use team filters to focus on key games
6. **Fire = strength:** Higher fire ratings = stronger picks
7. **Edge matters:** Bigger edge = more value vs market
8. **Confidence threshold:** Below 55% filtered out automatically

---

**Version:** NBA Model v33.0.8.0 (4 Markets: 1H Spread, 1H Total, FG Spread, FG Total)

**Last Updated:** 2025-12-31
