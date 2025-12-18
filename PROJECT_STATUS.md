# NBA v5.0 BETA - Project Status

## ✅ Docker-First Architecture

**All operations run through Docker containers.** No local Python execution required.

**Date:** December 2025  
**Status:** Production-ready containerized stack

---

## 🐳 Docker Stack

### Main Services (`docker compose up -d`)

| Service | Port | Status |
|---------|------|--------|
| `strict-api` | 8090 | ✅ Production ready - Main prediction API |
| `prediction-service` | 8082 | ✅ Working |
| `api-gateway` | 8080 | ✅ Working |
| `feature-store` | 8081 | ✅ Scaffolded |
| `line-movement-analyzer` | 8084 | ✅ Scaffolded |
| `schedule-poller` | 8085 | ✅ Scaffolded |
| `postgres` | 5432 | ✅ TimescaleDB |
| `redis` | 6379 | ✅ Working |

### Backtest Services (`docker compose -f docker-compose.backtest.yml`)

| Service | Purpose | Status |
|---------|---------|--------|
| `backtest-full` | Full pipeline | ✅ Working |
| `backtest-data` | Data only | ✅ Working |
| `backtest-only` | Backtest only | ✅ Working |
| `backtest-shell` | Debug shell | ✅ Working |

---

## 📊 Performance (Backtested)

| Market | Accuracy | ROI |
|--------|----------|-----|
| FG Spread | 60.6% | +15.7% |
| FG Total | 59.2% | +13.1% |
| FG Moneyline | 65.5% | +25.1% |
| 1H Spread | 55.9% | +8.2% |
| 1H Total | 58.1% | +11.4% |
| 1H Moneyline | 63.0% | +19.8% |

*316+ predictions tested (Oct-Dec 2025)*

---

## 🚀 Quick Commands

**Start stack:**
```powershell
docker compose up -d
```

**Check health:**
```powershell
curl http://localhost:8090/health
```

**Get predictions:**
```powershell
curl http://localhost:8090/slate/today
```

**Full analysis:**
```powershell
python scripts/analyze_slate_docker.py --date today
```

**Run backtest:**
```powershell
docker compose -f docker-compose.backtest.yml up backtest-full
```

**Stop stack:**
```powershell
docker compose down
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Main production stack |
| `docker-compose.backtest.yml` | Backtest stack |
| `Dockerfile` | Main API container |
| `Dockerfile.backtest` | Backtest container |
| `.env.example` | Environment template |

---

## ⚠️ Important Notes

1. **No local Python execution** - Everything runs in containers
2. **Use `analyze_slate_docker.py`** - Legacy scripts are disabled
3. **API keys required** - Set in `.env` before starting

---

## 📚 Documentation

- `README.md` - Full documentation
- `QUICK_START.md` - Getting started
- `docs/DOCKER_TROUBLESHOOTING.md` - Common issues
- `docs/CURRENT_STACK_AND_FLOW.md` - Architecture details
