# NBA V4.0 - Quick Reference Guide

## 🚀 Daily Workflow

### Option 1: Run Everything (Recommended)
```bash
python scripts/full_pipeline.py
```

### Option 2: Step-by-Step
```bash
# 1. Fetch today's odds
python scripts/run_the_odds_tomorrow.py

# 2. Get injury data
python scripts/fetch_injuries.py

# 3. Process the odds data
python scripts/process_odds_data.py

# 4. Generate predictions
python scripts/predict.py
```

---

## 📊 What Each Script Does

| Script | Purpose | Output | Time |
|--------|---------|--------|------|
| `run_the_odds_tomorrow.py` | Fetch ALL odds data | `data/raw/the_odds/YYYY-MM-DD/*.json` | ~30s |
| `fetch_injuries.py` | Get injury reports | `data/processed/injuries.csv` | ~5s |
| `process_odds_data.py` | Extract splits & lines | `data/processed/betting_splits.csv` | ~10s |
| `train_models.py` | Train base models | `data/processed/models/*.joblib` | ~2min |
| `train_ensemble_models.py` | Train ensemble models | `data/processed/models/*_ensemble.joblib` | ~1min |
| `predict.py` | Generate predictions | `data/processed/predictions.csv` | ~10s |
| `full_pipeline.py` | Run everything | All of the above | ~3-5min |

---

## 📁 Key Files & Locations

### Input Data
```
data/raw/the_odds/YYYY-MM-DD/
├── events_*.json              # All NBA events
├── sport_odds_*.json          # Odds for all games
├── event_{id}_odds_*.json     # Per-game detailed odds
└── event_{id}_markets_*.json  # All markets (props, 1H, etc)
```

### Processed Data
```
data/processed/
├── injuries.csv              # Current injuries
├── betting_splits.csv        # Line movement, RLM
├── first_half_lines.csv      # 1H lines (when available)
├── training_data.csv         # Historical training data
└── predictions.csv           # TODAY'S PREDICTIONS ← MAIN OUTPUT
```

### Models
```
data/processed/models/
├── spreads_model.joblib       # Base spreads model
├── totals_model.joblib        # Base totals model
├── spreads_ensemble.joblib    # Ensemble spreads (better!)
├── totals_ensemble.joblib     # Ensemble totals (better!)
└── manifest.json              # Model tracking
```

---

## 🎯 What Data Gets Collected

### From The Odds API
- ✅ All NBA games (events)
- ✅ Consensus lines (spreads, totals, moneylines)
- ✅ All bookmaker odds (FanDuel, DraftKings, etc)
- ✅ Line movement (opening → current)
- ✅ Alternative markets (1H, team totals, etc)

### From ESPN
- ✅ Injury reports (out, doubtful, questionable)
- ✅ Player stats (PPG when enriched)

### Calculated/Derived
- ✅ Line movement (-6.0 → -5.5 = +0.5)
- ✅ Bookmaker disagreement (std dev of lines)
- ✅ RLM detection (reverse line movement)
- ✅ Injury impact estimation

---

## 📈 Model Coverage

| Bet Type | Model Available | Prediction Ready | Data Source |
|----------|----------------|------------------|-------------|
| **Spreads** | ✅ Ensemble | ✅ Yes | Odds API + Injuries |
| **Totals** | ✅ Ensemble | ✅ Yes | Odds API + Injuries |
| **Moneyline** | ✅ Base | ✅ Yes | Odds API |
| **1H Spreads** | ✅ Base | ⚠️ Lines needed | Odds API markets |
| **1H Totals** | ✅ Base | ⚠️ Lines needed | Odds API markets |
| **Team Totals** | ✅ Base | ⚠️ Lines needed | Odds API markets |

---

## 🔧 Troubleshooting

### No games found
```bash
# Check what date script is looking for
python scripts/run_the_odds_tomorrow.py
# Look for: "Looking up ... for TOMORROW: YYYY-MM-DD"
```

### API Key issues
```bash
# Check if key is set
echo $env:THE_ODDS_API_KEY

# Set it (PowerShell)
$env:THE_ODDS_API_KEY = 'your_key_here'

# Or add to .env file
echo "THE_ODDS_API_KEY=your_key_here" >> .env
```

### No training data
```bash
# Generate it first
python scripts/generate_training_data.py
```

### Models not found
```bash
# Train them
python scripts/train_models.py
python scripts/train_ensemble_models.py
```

---

## 📊 Reading the Output

### predictions.csv
```csv
date,home_team,away_team,predicted_spread,confidence,predicted_total,home_ppg,away_ppg
2024-12-04 19:00 CST,Lakers,Celtics,-5.5,0.64,219.8,115.2,118.5
```

**Interpretation**:
- Game: Celtics @ Lakers at 7pm CST
- Model predicts: **Celtics +5.5** with 64% confidence
- Total prediction: **Under 219.8** (line probably ~222.5)
- Reasoning: Celtics averaging 118.5 PPG vs Lakers 115.2 PPG

### betting_splits.csv
```csv
event_id,home_team,spread_line,spread_open,spread_current,spread_movement
abc123,Lakers,-5.5,-6.0,-5.5,+0.5
```

**Interpretation**:
- Line opened: Lakers -6.0
- Line now: Lakers -5.5
- Movement: +0.5 toward Celtics
- **Possible sharp money on Celtics** (line moving against Lakers)

---

## 🎯 Best Practices

### Daily Routine
1. Morning: Run `python scripts/run_the_odds_tomorrow.py`
2. Check: Review `data/processed/predictions.csv`
3. Compare: Check current lines vs model predictions
4. Bet: Take positions with 5%+ edge and 60%+ confidence

### Weekly Maintenance
1. Run full pipeline: `python scripts/full_pipeline.py`
2. Check model performance in manifest.json
3. Retrain if accuracy drops below 55%

### Monthly Updates
1. Generate fresh training data
2. Train new models with updated data
3. Compare old vs new model performance
4. Promote better model to production

---

## 💡 Pro Tips

### Finding Value Bets
```python
# Model says 64% chance Celtics cover
# Line implies 52% chance (from odds)
# Edge = 64% - 52% = 12% edge ← BET THIS!
```

### Using Injury Data
```python
# LeBron James out (25 PPG)
# Model accounts for this in prediction
# If line hasn't moved enough → VALUE
```

### Line Movement
```python
# Line moved FROM Lakers -6.0 TO -5.5
# But public usually on favorites (Lakers)
# Line should move TO -6.5 if public on Lakers
# Reverse movement = SHARP MONEY on Celtics
```

---

## 🔗 Related Files

- `IMPLEMENTATION_SUMMARY.md` - Complete feature list
- `DATA_FLOW_GUIDE.md` - Detailed data flow breakdown
- `README.md` - Project overview
- `src/ingestion/README.md` - API integration details

---

## 🆘 Getting Help

1. Check the error message
2. Review relevant guide (this file, DATA_FLOW_GUIDE.md)
3. Verify API keys are set
4. Check data files exist in expected locations
5. Run scripts individually to isolate issue

---

**Generated**: 2024-12-03
**Version**: 4.1.0
