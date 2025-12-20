# NBA v5.0 BETA - Single Source of Truth

**Master Document** - All authoritative information for the system.

---

## 🚀 THE ONE COMMAND

```powershell
.\run.ps1
```

This single command does everything:
1. Starts the production container via `docker compose up -d` (builds if needed)
2. Waits for the API to be ready
3. Runs analysis for the requested date/game(s) and saves reports to `data/processed/`
4. Displays picks with fire ratings

**Options:**
```powershell
.\run.ps1 -Date tomorrow                          # Tomorrow's games
.\run.ps1 -Matchup "Lakers"                       # Filter to specific team
.\run.ps1 -Date today -Matchup "Lakers vs Celtics" # Specific game
.\run.ps1 -Date 2025-12-19 -Matchup "Lakers vs Celtics, Heat @ Knicks" # Multiple filters
```

---

## 🎯 Entry Points

| What | Command/URL | Purpose |
|------|-------------|---------|
| **Run Predictions** | `.\run.ps1` | **THE ONE COMMAND** - Starts container and runs analysis |
| **Main API** | `http://localhost:8090` | Direct API access |
| **Health Check** | `http://localhost:8090/health` | Verify system is running |
| **Stop Container** | `docker compose down` | Stop the production container |
| **Cleanup Docker** | `.\scripts\cleanup_nba_docker.ps1 -All -Force` | Remove old containers/images/volumes |
| **Run Backtest** | `docker compose -f docker-compose.backtest.yml up backtest-full` | Full backtest (dev only) |

---

## 📊 The 6 Backtested Markets

| Market | Accuracy | ROI | Status |
|--------|----------|-----|--------|
| FG Spread | 60.6% | +15.7% | ✅ Production |
| FG Total | 59.2% | +13.1% | ✅ Production |
| FG Moneyline | 65.5% | +25.1% | ✅ Production |
| 1H Spread | 55.9% | +8.2% | ✅ Production |
| 1H Total | 58.1% | +11.4% | ✅ Production |
| 1H Moneyline | 63.0% | +19.8% | ✅ Production |

*Validated on 316+ predictions (Oct-Dec 2025)*

---

## 🐳 Docker Architecture

### Production (Single Container)

| Component | Details |
|-----------|---------|
| **Image** | `nba-strict-api:latest` |
| **Container** | `nba-api` (port 8090:8080) |
| **Models** | Baked into image from `models/production/` |
| **Dependencies** | None (no database/microservices) |
| **Entry Point** | `.\run.ps1` (wrapper) / `python scripts/run_slate.py` (source of truth) |

### Backtest Stack (docker-compose.backtest.yml)

| Service | Command | Purpose |
|---------|---------|---------|
| backtest-full | `up backtest-full` | Full pipeline |
| backtest-data | `up backtest-data` | Data only |
| backtest-only | `up backtest-only` | Backtest only |
| backtest-shell | `run --rm backtest-shell` | Debug shell |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `Dockerfile` | **PRODUCTION IMAGE** - Single container with baked-in models |
| `run.ps1` | **PRODUCTION ENTRY POINT** - Builds and runs the container |
| `models/production/` | **PRODUCTION MODEL PACK** - Backtested models (baked into image) |
| `docker-compose.backtest.yml` | Backtest pipeline (development only) |
| `Dockerfile.backtest` | Backtest container |
| `.env.example` | Environment template |
| `src/serving/app.py` | **MAIN API CODE** |
| `src/prediction/engine.py` | Unified prediction engine |
| `scripts/analyze_slate_docker.py` | Analysis script (calls container API) |
| `scripts/cleanup_nba_docker.ps1` | Cleanup old Docker resources |

---

## 🔑 Required API Keys

```env
THE_ODDS_API_KEY=xxx     # The Odds API (required)
API_BASKETBALL_KEY=xxx   # API-Basketball (required)
```

Optional:
```env
ACTION_NETWORK_USERNAME=xxx
ACTION_NETWORK_PASSWORD=xxx
BETSAPI_KEY=xxx
```

---

## 📡 API Endpoints

### GET /health
```json
{"status": "ok", "mode": "STRICT", "markets": 6, "engine_loaded": true}
```

### GET /slate/{date}
Get predictions for a date (`today`, `tomorrow`, or `YYYY-MM-DD`).

### GET /slate/{date}/comprehensive
Full analysis with edges, rationale, and betting splits.

### POST /predict/game
Single game prediction (requires all 8 line parameters).

---

## 🚀 Daily Workflow

```powershell
# 1. Run predictions (starts container if needed, runs analysis)
.\run.ps1

# Or with options:
.\run.ps1 -Date tomorrow
.\run.ps1 -Matchup "Lakers"

# 2. View results in data/processed/slate_analysis_*.txt

# 3. Stop container when done (optional)
docker compose down
```

---

## 📂 Directory Structure

```
nba_v5.0_BETA/
├── docker-compose.yml         # PRODUCTION STACK
├── docker-compose.backtest.yml
├── Dockerfile                 # strict-api
├── Dockerfile.backtest
├── .env.example
│
├── src/                       # Python source
│   ├── serving/app.py         # MAIN API
│   ├── prediction/engine.py   # Prediction engine
│   ├── modeling/              # Models & features
│   └── ingestion/             # Data sources
│
├── scripts/
│   ├── run_slate.py             # SINGLE SOURCE OF TRUTH runner (starts compose + calls API)
│   └── analyze_slate_docker.py  # Legacy/alternate formatter (calls API)
│
├── docs/                      # Documentation
│
├── data/
│   ├── processed/             # Models, predictions
│   └── results/               # Backtest results
│
└── docs/                      # Documentation
    ├── SINGLE_SOURCE_OF_TRUTH.md  # THIS FILE
    ├── CURRENT_STACK_AND_FLOW.md
    ├── BACKTEST_STATUS.md
    ├── NEXT_STEPS.md
    ├── DATA_INGESTION_METHODOLOGY.md
    ├── DATA_SOURCE_OF_TRUTH.md
    └── DOCKER_TROUBLESHOOTING.md
```

---

## ⚠️ Important Rules

1. **Single Container Production** - One hardened Docker image with all models baked-in (`nba-strict-api:latest`)
2. **No External Dependencies** - Production container has no database/microservice dependencies
3. **STRICT MODE** - All inputs required, no fallbacks
4. **6 Markets Only** - Only backtested markets (no Q1)
5. **Models in Repo** - Production models committed in `models/production/` for reproducibility
6. **Fail Loudly** - Errors crash, not silently pass

---

## 📚 Other Docs

| Doc | Purpose |
|-----|---------|
| `CURRENT_STACK_AND_FLOW.md` | Detailed architecture |
| `BACKTEST_STATUS.md` | Backtest results |
| `NEXT_STEPS.md` | What to do next |
| `DATA_INGESTION_METHODOLOGY.md` | Data sources |
| `DATA_SOURCE_OF_TRUTH.md` | Data policies |
| `DOCKER_TROUBLESHOOTING.md` | Debug help |

---

## ✅ Quick Verification

```powershell
# Is the stack running?
docker ps --filter "name=nba"

# Is the API healthy?
curl http://localhost:8090/health

# Are models loaded?
# Response should show: "engine_loaded": true
```
