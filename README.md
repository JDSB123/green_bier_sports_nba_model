# NBA Basketball Prediction System v5.0 BETA

Production-grade microservices architecture for NBA betting predictions, refactored from v4.0 monolith.

## Architecture Overview

This version uses a **microservices architecture** with Go, Rust, and Python services, matching the approach used in `ncaam_v5.0_BETA`.

### Technology Stack

| Service | Language | Purpose |
|---------|----------|---------|
| Odds Ingestion | Rust | Real-time odds streaming (<10ms latency) |
| Schedule Poller | Go | Game schedule aggregation |
| Feature Store | Go | High-performance feature serving |
| Prediction Service | Python | ML inference (NBA v4.0 models) |
| Line Movement Analyzer | Go | RLM detection and line movement analysis |
| API Gateway | Go | Unified REST API |

### Infrastructure

- **PostgreSQL 15 + TimescaleDB** (time-series data)
- **Redis 7** (caching, pub/sub)
- **Docker Compose** (dev) / **Kubernetes** (prod)

## Project Structure

```
nba_v5.0_BETA/
├── services/
│   ├── odds-ingestion-rust/       # Rust: Real-time odds
│   ├── schedule-poller-go/       # Go: Game schedules
│   ├── feature-store-go/          # Go: Feature serving
│   ├── prediction-service-python/ # Python: ML inference
│   ├── line-movement-analyzer-go/ # Go: Line movement analysis
│   └── api-gateway-go/            # Go: Unified API
├── database/                      # SQL migrations
├── docker-compose.yml            # Local development
└── README.md                      # This file
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Go 1.22+ (for local Go development)
- Rust 1.75+ (for local Rust development)
- Python 3.11+ (for local Python development)

### Environment Setup

1. **Copy environment template:**
   ```powershell
   copy .env.example .env
   ```

2. **Fill in API keys in `.env`:**
   ```env
   THE_ODDS_API_KEY=your_key_here
   API_BASKETBALL_KEY=your_key_here
   DB_PASSWORD=nba_dev_password
   ```

### Running Services

**Start all services:**
```powershell
docker-compose up -d
```

**Check service health:**
```powershell
curl http://localhost:8080/health
curl http://localhost:8082/health  # Prediction service
curl http://localhost:8081/health  # Feature store
```

**View logs:**
```powershell
docker-compose logs -f prediction-service
docker-compose logs -f odds-ingestion
```

**Stop all services:**
```powershell
docker-compose down
```

## API Usage

### Get Predictions

```bash
POST http://localhost:8080/api/predict
Content-Type: application/json

{
  "game_id": "123e4567-e89b-12d3-a456-426614174000",
  "home_team": "Lakers",
  "away_team": "Celtics",
  "commence_time": "2025-12-18T19:30:00Z",
  "features": {
    "home_ppg": 112.5,
    "home_papg": 108.2,
    "away_ppg": 118.7,
    "away_papg": 115.3,
    "predicted_margin": -6.2,
    "predicted_total": 223.5
  },
  "market_odds": {
    "spread": -6.5,
    "total": 223.5,
    "home_ml": -280,
    "away_ml": +230
  }
}
```

### Get Features

```bash
GET http://localhost:8080/api/features?team=Lakers&date=2025-12-18
```

### Get Odds

```bash
GET http://localhost:8080/api/odds?date=2025-12-18
```

## Development

### Local Development (Without Docker)

**Go Services:**
```powershell
cd services/api-gateway-go
go run main.go
```

**Rust Service:**
```powershell
cd services/odds-ingestion-rust
cargo run
```

**Python Service:**
```powershell
cd services/prediction-service-python
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8082
```

## Migration from v4.0

This v5.0 BETA is a **refactored microservices version** of the v4.0 monolith. Key differences:

- **Architecture**: Microservices vs monolith
- **Languages**: Go + Rust + Python vs Python only
- **Deployment**: Docker containers vs single process
- **Scalability**: Independent service scaling vs single process

The **prediction logic remains the same** - we're using the same ML models and algorithms from v4.0, just distributed across services.

## Next Steps

1. **Complete prediction service integration** - Connect to actual NBA v4.0 models
2. **Implement feature store** - Real feature computation from NBA v4.0
3. **Complete odds ingestion** - Full database/Redis integration
4. **Add monitoring** - Prometheus + Grafana
5. **Production deployment** - Kubernetes manifests

## Status

⚠️ **BETA** - This is a work in progress. Core services are scaffolded but need full implementation.

## Reference

- **NBA v4.0**: Original monolith (production-ready)
- **ncaam_v5.0_BETA**: Reference microservices architecture
