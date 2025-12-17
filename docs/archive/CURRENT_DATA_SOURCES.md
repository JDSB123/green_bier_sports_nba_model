# Current NBA Data Sources

**Version:** 4.0.0 (Streamlined)
**Last Updated:** 2025-12-06

---

## 🎯 Active Data Sources (4)

| # | Module | API/Source | Purpose | Status | Cost |
|---|--------|-----------|---------|--------|------|
| 1 | **the_odds.py** | The Odds API | Betting lines & odds | ✅ Primary | Paid |
| 2 | **api_basketball.py** | API-Basketball | Team statistics & standings | ✅ Active | Paid |
| 3 | **injuries.py** | ESPN | Injury reports | ✅ Active | FREE |
| 4 | **betting_splits.py** | SBRO/Covers/Mock | Public betting %s | ✅ Active | FREE |

---

## 📊 Data Flow

```
1. THE ODDS API
   └─→ Spreads, Totals, Moneylines, Line Movement
       └─→ 10+ bookmakers consensus
           └─→ data/raw/the_odds/

2. API-BASKETBALL
   └─→ Team season stats (PPG, PAPG, Win%, etc.)
       └─→ H2H history, Standings, Recent games
           └─→ data/raw/api_basketball/ + cache/

3. ESPN INJURIES
   └─→ Player status (Out, Doubtful, Questionable)
       └─→ Injury type, Expected return
           └─→ data/raw/injuries/

4. BETTING SPLITS
   └─→ Public betting percentages (tickets & money)
       └─→ RLM detection, Sharp money indicators
           └─→ data/raw/betting_splits/
```

---

## 🔧 Collection Scripts

| Script | Purpose | Frequency | Command |
|--------|---------|-----------|---------|
| `run_the_odds_tomorrow.py` | Fetch odds for tomorrow's games | 3x daily | `python scripts/run_the_odds_tomorrow.py` |
| `collect_api_basketball.py` | Fetch team statistics | Weekly | `python scripts/collect_api_basketball.py` |
| `fetch_injuries.py` | Fetch injury reports | 2x daily | `python scripts/fetch_injuries.py` |
| `collect_betting_splits.py` | Fetch public betting %s | 2x daily | `python scripts/collect_betting_splits.py --save` |
| `ingest_all.py` | **Orchestrate all** | Daily | `python scripts/ingest_all.py` |

---

## ⚙️ Required Environment Variables

```bash
# .env file
THE_ODDS_API_KEY=your_key_here
API_BASKETBALL_KEY=your_key_here

# Optional (for advanced features)
ACTION_NETWORK_API_KEY=your_key_here  # For real betting splits
```

---

## 📈 Features Generated

### From The Odds API (10 features)
- `spread_line` - Consensus spread
- `total_line` - Consensus total
- `spread_opening_line` - Opening line
- `spread_movement` - Line movement magnitude
- `spread_line_std` - Bookmaker disagreement
- `best_home_line` - Best available line for home
- `best_away_line` - Best available line for away
- `total_opening_line` - Opening total
- `total_movement` - Total line movement
- `total_line_std` - Total disagreement

### From API-Basketball (15+ features)
- `home_ppg`, `away_ppg` - Points per game
- `home_papg`, `away_papg` - Points allowed
- `home_win_pct`, `away_win_pct` - Win percentages
- `home_position`, `away_position` - Standings position
- `h2h_win_rate` - Head-to-head history
- `home_elo`, `away_elo` - ELO ratings (derived)
- `predicted_margin` - Model prediction
- `predicted_total` - Total prediction

### From ESPN Injuries (8 features)
- `home_injury_spread_impact` - Point impact of injuries
- `home_injury_total_impact` - Total impact
- `home_players_out` - Count of missing players
- `home_star_out` - Star player missing flag
- `away_injury_spread_impact`
- `away_injury_total_impact`
- `away_players_out`
- `away_star_out`

### From Betting Splits (14 features) ✨ NEW
- `spread_public_home_pct` - % tickets on home
- `spread_money_home_pct` - % money on home
- `over_public_pct` - % tickets on over
- `over_money_pct` - % money on over
- `spread_ticket_money_diff` - Divergence (sharp signal)
- `total_ticket_money_diff` - Total divergence
- `is_rlm_spread` - RLM detected (binary)
- `is_rlm_total` - Total RLM detected
- `sharp_side_spread` - 1 (home), -1 (away), 0 (neutral)
- `sharp_side_total` - 1 (over), -1 (under), 0 (neutral)
- Plus additional line movement features

---

## 🚀 Quick Start

### 1. Collect All Data
```bash
python scripts/ingest_all.py
```

### 2. Generate Predictions
```bash
python scripts/predict.py --date tomorrow
```

### 3. View Predictions
```bash
cat data/processed/predictions.csv
```

---

## 🔍 Data Quality Checks

### The Odds API
- ✅ 10+ bookmakers for consensus
- ✅ Real-time line movement tracking
- ✅ CST timezone aware

### API-Basketball
- ✅ Official NBA statistics
- ✅ Daily updates during season
- ✅ Cached for performance

### ESPN Injuries
- ✅ Official injury reports
- ✅ Updated twice daily
- ✅ No API key required

### Betting Splits
- ✅ Auto-fallback (SBRO → Covers → Mock)
- ✅ RLM detection algorithm
- ✅ Realistic mock data for development

---

## 📚 Documentation

- `DATA_INGESTION_ARCHITECTURE.md` - Complete architecture details
- `REMOVED_MODULES_SUMMARY.md` - What was removed and why
- `CLEANUP_AND_INTEGRATION_SUMMARY.md` - Recent enhancements
- `QUICK_REFERENCE.md` - Command reference

---

## ✅ Verification

**Test imports:**
```bash
python -c "from src.ingestion import the_odds, injuries, betting_splits; from src.ingestion.api_basketball import APIBasketballClient; print('OK')"
```

**Test ingestion:**
```bash
python scripts/ingest_all.py --essential
```

**Test predictions:**
```bash
python scripts/predict.py --date tomorrow
```

---

**Current State:** Clean, focused, 4-source data pipeline with no redundancy.
