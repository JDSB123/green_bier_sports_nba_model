# System Architecture

**Last Updated:** 2026-01-23
**Status:** Consolidated from ARCHITECTURE_FLOW_AND_ENDPOINTS, STACK_FLOW_AND_VERIFICATION

---

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (port 8090)                   │
│                    src/serving/app.py                           │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                     │
│  - /health           Health check                               │
│  - /slate/{date}     Get predictions for date                   │
│  - /slate/{date}/executive    Executive summary                 │
│  - /predict/game     Single game prediction                     │
│  - /markets          List available markets                     │
│  - /verify           Verification check                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                UnifiedPredictionEngine                          │
│                src/prediction/engine.py                         │
├─────────────────────────────────────────────────────────────────┤
│  Markets: 1h_spread, 1h_total, fg_spread, fg_total             │
│  Models: 4 independent scikit-learn models                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                RichFeatureBuilder                               │
│                src/features/rich_features.py                    │
├─────────────────────────────────────────────────────────────────┤
│  Fetches: Odds, Stats, Injuries, Betting Splits                │
│  Computes: All prediction features in real-time                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                Data Ingestion Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  The Odds API    → Betting lines, odds                         │
│  API-Basketball  → Game outcomes, team stats                    │
│  ESPN            → Schedules, injuries                          │
│  Action Network  → Betting splits (optional)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check, version info |
| `/slate/{date}` | GET | Predictions for a date |
| `/slate/{date}/executive` | GET | Executive summary |
| `/slate/{date}/comprehensive` | GET | Full detailed predictions |
| `/predict/game` | POST | Single game prediction |
| `/markets` | GET | Available markets list |
| `/verify` | GET | System verification |

### Example: Get Today's Slate

```bash
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/slate/today"
```

Response:
```json
{
  "date": "2026-01-23",
  "games": [...],
  "picks": [...],
  "summary": {...}
}
```

---

## Data Flow

### Training Data Pipeline

```
Raw APIs → Standardization → Merge → Labels → Features → training_data.csv
                                                              │
                                                              ▼
                                                         Model Training
                                                              │
                                                              ▼
                                                      models/production/
```

### Prediction Pipeline

```
Request → Fetch Live Data → Build Features → Model Predict → Filter → Response
```

---

## Ingestion Endpoints

### The Odds API

| Endpoint | Purpose |
|----------|---------|
| `/sports` | List sports |
| `/sports/.../events` | Game events |
| `/sports/.../odds` | Betting odds |
| `/events/{id}/odds` | Event-specific odds (1H) |
| `/historical/sports/.../odds` | Historical odds |

### API-Basketball

| Endpoint | Purpose |
|----------|---------|
| `/teams` | Team data |
| `/games` | Game results with Q1-Q4 scores |
| `/statistics` | Team statistics |
| `/games/statistics/teams` | Advanced team stats |

---

## Container Startup

```
/app/start.sh (Dockerfile.combined)
    │
    └─ uvicorn src.serving.app:app --port 8090
```

---

## Feature Engineering

### Step 1: Fetch Data

- Game schedules from The Odds API
- Team stats from API-Basketball
- Injuries from ESPN + API-Basketball
- Betting splits from Action Network

### Step 2: Standardize

All team names normalized to canonical format.

### Step 3: Compute Features

- Rolling stats (PPG, PAPG, margins)
- Rest days and travel
- Injury impact
- Elo ratings
- Betting market data

### Step 4: Model Prediction

Features → XGBoost models → Predicted margins/totals

### Step 5: Filter

Apply edge-only thresholds (confidence is informational only).

---

## Azure Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  nba-gbsv-model-rg                              │
├─────────────────────────────────────────────────────────────────┤
│  Container App: nba-gbsv-api                                    │
│  Container Registry: nbagbsacr                                  │
│  Key Vault: nbagbs-keyvault                                     │
│  Storage: nbagbsvstrg                                           │
│  Log Analytics: gbs-logs-prod                                   │
│  App Insights: gbs-insights-prod                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Health Check Response

```json
{
  "status": "ok",
  "version": "NBA_v33.1.8",
  "mode": "STRICT",
  "markets": 4,
  "markets_list": ["1h_spread", "1h_total", "fg_spread", "fg_total"],
  "engine_loaded": true,
  "api_keys": {
    "THE_ODDS_API_KEY": "set",
    "API_BASKETBALL_KEY": "set"
  }
}
```
