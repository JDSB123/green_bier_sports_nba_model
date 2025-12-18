# NBA v5.0 BETA - Single Source of Truth

**Master Document** - All authoritative information for the system.

---

## 🚀 THE ONE COMMAND

```powershell
python scripts/run_slate.py
```

This single command does everything:
1. Checks Docker is running
2. Starts the stack if needed
3. Waits for API to be healthy
4. Fetches predictions
5. Displays picks with fire ratings

**Options:**
```powershell
python scripts/run_slate.py --date tomorrow        # Tomorrow's games
python scripts/run_slate.py --matchup Lakers       # Filter to specific team
python scripts/run_slate.py --date 2025-12-19 --matchup Celtics
```

---

## 🎯 Entry Points

| What | Command/URL | Purpose |
|------|-------------|---------|
| **Run Predictions** | `python scripts/run_slate.py` | **THE ONE COMMAND** |
| **Main API** | `http://localhost:8090` | Direct API access |
| **Health Check** | `http://localhost:8090/health` | Verify system is running |
| **Stop Stack** | `docker compose down` | Stop all services |
| **Run Backtest** | `docker compose -f docker-compose.backtest.yml up backtest-full` | Full backtest |

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

## 🐳 Docker Services

### Production Stack (docker-compose.yml)

| Service | Port | Status | Purpose |
|---------|------|--------|---------|
| **strict-api** | 8090 | ✅ PRIMARY | Main prediction API - USE THIS |
| prediction-service | 8082 | ✅ | ML inference (internal) |
| api-gateway | 8080 | ✅ | REST gateway (scaffolded) |
| feature-store | 8081 | 🚧 | Feature serving (scaffolded) |
| line-movement-analyzer | 8084 | 🚧 | RLM detection (scaffolded) |
| schedule-poller | 8085 | 🚧 | Schedule aggregation (scaffolded) |
| postgres | 5432 | ✅ | TimescaleDB |
| redis | 6379 | ✅ | Cache |

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
| `docker-compose.yml` | **PRODUCTION STACK** - Start here |
| `docker-compose.backtest.yml` | Backtest pipeline |
| `Dockerfile` | strict-api container |
| `Dockerfile.backtest` | Backtest container |
| `.env.example` | Environment template |
| `src/serving/app.py` | **MAIN API CODE** |
| `src/prediction/engine.py` | Unified prediction engine |
| `scripts/analyze_slate_docker.py` | Docker-only analysis script |

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
# 1. Start stack
docker compose up -d

# 2. Verify health
curl http://localhost:8090/health

# 3. Get analysis
python scripts/analyze_slate_docker.py --date today

# 4. View results in data/processed/slate_analysis_*.txt
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
│   └── analyze_slate_docker.py  # Docker-only analysis
│
├── services/                  # Microservices (Go/Rust)
│   ├── api-gateway-go/
│   ├── feature-store-go/
│   ├── line-movement-analyzer-go/
│   ├── odds-ingestion-rust/
│   ├── prediction-service-python/
│   └── schedule-poller-go/
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

1. **Docker Only** - No local Python execution
2. **STRICT MODE** - All inputs required, no fallbacks
3. **6 Markets Only** - Only backtested markets (no Q1)
4. **No Placeholders** - Real data only, no mocks
5. **Fail Loudly** - Errors crash, not silently pass

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
