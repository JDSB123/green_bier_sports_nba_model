# Current Stack & Ingestion Flow

**Date:** 2025-12-17  
**Status:** Hybrid Architecture (v4.0 Monolith + v5.0 BETA Microservices)

---

## How We Got ROI Numbers

### The Backtest (Oct 2 - Dec 9, 2025)

The ROI numbers (60.6% spreads, 59.2% totals) came from a **backtest that was run earlier** when models existed:

1. **Models were trained** using training data (6,290 games from 2010-2025)
2. **Backtest was run** on 422 games from the 2025-2026 season (Oct 2 - Dec 9)
3. **Results documented** in `docs/BACKTEST_RESULTS_SUMMARY.md`:
   - Spreads: 60.6% accuracy, +15.7% ROI (with filtering)
   - Totals: 59.2% accuracy, +13.1% ROI (baseline)

### Current State

- **Models:** Not currently on disk (may have been deleted/moved)
- **Manifest:** Shows models were last trained Dec 17, 2025 at 18:48:55
- **Training Data:** Still exists (`data/processed/training_data.csv` - 6,290 games)
- **Backtest Results:** Documented but models need to be retrained to reproduce

**To reproduce ROI:**
```bash
# 1. Retrain models (uses existing training data)
python scripts/train_models.py

# 2. Run backtest (requires models + betting lines)
python scripts/backtest.py --markets all
```

---

## Current Architecture: Hybrid Stack

### v4.0 Monolith (Production-Ready) ✅

**Status:** Fully functional, production-ready Python monolith

**Components:**
- **Data Ingestion:** Python scripts in `src/ingestion/`
- **Model Training:** Python scripts in `scripts/`
- **Predictions:** Python scripts in `scripts/predict.py`
- **Storage:** CSV files in `data/raw/` and `data/processed/`

**Technology:**
- Python 3.11+
- scikit-learn (models)
- pandas (data processing)
- httpx/requests (API clients)

### v5.0 BETA Microservices (In Development) 🚧

**Status:** Scaffolded but not fully implemented

**Services:**
- `odds-ingestion-rust/` - Real-time odds streaming
- `schedule-poller-go/` - Game schedule aggregation
- `feature-store-go/` - Feature serving
- `prediction-service-python/` - ML inference
- `line-movement-analyzer-go/` - RLM detection
- `api-gateway-go/` - Unified REST API

**Infrastructure:**
- PostgreSQL 15 + TimescaleDB
- Redis 7
- Docker Compose

**Note:** These services are scaffolded but need full implementation to connect to v4.0 models.

---

## Current Ingestion Flow (v4.0 Monolith)

### Data Sources

| Source | Module | Purpose | Status |
|--------|--------|---------|--------|
| **The Odds API** | `src/ingestion/the_odds.py` | Live betting odds, spreads, totals, moneyline | ✅ Active |
| **API-Basketball** | `src/ingestion/api_basketball.py` | Game scores, team stats, standings, H2H | ✅ Active |
| **ESPN** | `src/ingestion/injuries.py` | Injury reports (free) | ✅ Active |
| **Betting Splits** | `src/ingestion/betting_splits.py` | Public betting percentages, RLM detection | ✅ Active |

### Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. The Odds API                                            │
│     └─→ scripts/run_the_odds_tomorrow.py                   │
│         • Live odds for upcoming games                      │
│         • Spreads, totals, moneyline                         │
│         • First-half markets                                │
│         • Output: data/raw/the_odds/YYYY-MM-DD/*.json      │
│                                                              │
│  2. API-Basketball                                          │
│     └─→ scripts/collect_api_basketball.py                   │
│         • Team statistics (PPG, PAPG, W-L)                │
│         • Game outcomes (scores, box scores)                │
│         • Head-to-head history                              │
│         • Output: data/raw/api_basketball/*.json           │
│                                                              │
│  3. ESPN Injuries                                           │
│     └─→ scripts/fetch_injuries.py                          │
│         • Player injury status                              │
│         • Impact assessment (PPG, minutes)                 │
│         • Output: data/processed/injuries.csv             │
│                                                              │
│  4. Betting Splits                                          │
│     └─→ scripts/collect_betting_splits.py                  │
│         • Public betting percentages                        │
│         • Line movement tracking                            │
│         • RLM (Reverse Line Movement) detection            │
│         • Output: data/processed/betting_splits.csv        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  STANDARDIZATION LAYER                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  src/ingestion/standardize.py                               │
│    • Team names → ESPN format                               │
│    • Date normalization                                      │
│    • Validation flags (_data_valid, _home_team_valid, etc.)│
│    • No fake data policy (rejects invalid data)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA PROCESSING                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. scripts/process_odds_data.py                            │
│     • Extract betting splits                                │
│     • Detect line movement                                  │
│     • Extract first-half lines                              │
│     • Output: betting_splits.csv, first_half_lines.csv     │
│                                                              │
│  2. scripts/build_training_dataset.py                       │
│     • Link odds → outcomes                                  │
│     • Feature engineering                                   │
│     • Rolling statistics                                    │
│     • Output: training_data.csv                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  scripts/train_models.py                                   │
│    • Train spreads model                                    │
│    • Train totals model                                     │
│    • Train moneyline model                                  │
│    • Output: data/processed/models/*.joblib                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    PREDICTIONS                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  scripts/predict.py                                         │
│    • Fetch upcoming games                                   │
│    • Build features                                         │
│    • Generate predictions (spreads, totals, ML)              │
│    • Apply smart filtering                                  │
│    • Output: predictions.csv, betting_card.csv             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Daily Workflow

### Automated Pipeline

```bash
# Run complete pipeline (recommended)
python scripts/full_pipeline.py

# Steps:
# 1. Fetch odds (The Odds API)
# 2. Fetch injuries (ESPN)
# 3. Process odds data
# 4. Archive cache
# 5. Build training dataset
# 6. Train models (optional: --skip-train)
# 7. Generate predictions
# 8. Review previous predictions
```

### Manual Steps

```bash
# 1. Fetch odds for tomorrow
python scripts/run_the_odds_tomorrow.py

# 2. Fetch injuries
python scripts/fetch_injuries.py

# 3. Process odds
python scripts/process_odds_data.py

# 4. Build training data (if needed)
python scripts/build_training_dataset.py

# 5. Train models (if needed)
python scripts/train_models.py

# 6. Generate predictions
python scripts/predict.py --date today
```

---

## Data Storage Structure

```
data/
├── raw/                          # Raw API responses
│   ├── the_odds/                 # The Odds API JSON files
│   │   └── YYYY-MM-DD/
│   ├── api_basketball/           # API-Basketball JSON files
│   └── injuries/                 # ESPN injury data
│
└── processed/                    # Processed/standardized data
    ├── training_data.csv         # Linked odds + outcomes (6,290 games)
    ├── betting_splits.csv       # Line movement, RLM
    ├── injuries.csv              # Standardized injury reports
    ├── predictions.csv           # Generated predictions
    ├── betting_card.csv          # Filtered plays (ready to bet)
    └── models/                   # Trained models
        ├── spreads_model.joblib
        ├── totals_model.joblib
        ├── moneyline_model.joblib
        └── manifest.json          # Model metadata
```

---

## Key Scripts

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `full_pipeline.py` | Complete end-to-end pipeline | Daily (automated) |
| `run_the_odds_tomorrow.py` | Fetch odds for upcoming games | 2-3x daily |
| `fetch_injuries.py` | Fetch injury reports | 2x daily |
| `process_odds_data.py` | Process odds into betting splits | After odds fetch |
| `build_training_dataset.py` | Build training data from raw data | Weekly or when new data |
| `train_models.py` | Train prediction models | Weekly or when new training data |
| `predict.py` | Generate predictions | Daily (before games) |
| `review_predictions.py` | Grade picks vs results | After games complete |

---

## API Keys Required

```env
# Required
THE_ODDS_API_KEY=your_key_here          # The Odds API (paid)
API_BASKETBALL_KEY=your_key_here        # API-Basketball (paid)

# Optional
BETSAPI_KEY=your_key_here               # BETSAPI (optional)
ACTION_NETWORK_USERNAME=your_username   # Action Network (optional)
ACTION_NETWORK_PASSWORD=your_password   # Action Network (optional)
KAGGLE_API_TOKEN=your_token             # Kaggle (optional, for historical data)
```

---

## Summary

**Current State:**
- ✅ **v4.0 Monolith:** Production-ready, fully functional
- 🚧 **v5.0 BETA Microservices:** Scaffolded, needs implementation
- ✅ **Data Ingestion:** Working (4 active sources)
- ⚠️ **Models:** Need to be retrained (files not on disk)
- ✅ **Training Data:** Available (6,290 games)
- ✅ **ROI Numbers:** From previous backtest (Oct-Dec 2025)

**To Get Production-Ready:**
1. Retrain models: `python scripts/train_models.py`
2. Run validation: `python scripts/validate_production_current.py`
3. Generate predictions: `python scripts/predict.py`

**Architecture Decision:**
- Use **v4.0 monolith** for production (fully working)
- Complete **v5.0 microservices** for future scalability (optional)

