# NBA v5.0 BETA - Quick Start Guide

## ✅ Setup Complete!

Your `.env` file has been created with all API keys configured.

## 🚀 Getting Started

### Option 1: Docker Compose (Recommended for Microservices)

```powershell
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8080/health
curl http://localhost:8082/health  # Prediction service
curl http://localhost:8081/health  # Feature store

# View logs
docker-compose logs -f prediction-service
docker-compose logs -f odds-ingestion

# Stop services
docker-compose down
```

### Option 2: Python Monolith (Original v4.0 approach)

If you want to use the original Python scripts while developing the microservices:

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run predictions
python scripts/predict.py

# Run full pipeline
python scripts/full_pipeline.py
```

## 📁 Project Structure

```
nba_v5.0_BETA/
├── services/              # Microservices (Go, Rust, Python)
│   ├── api-gateway-go/
│   ├── feature-store-go/
│   ├── line-movement-analyzer-go/
│   ├── schedule-poller-go/
│   ├── odds-ingestion-rust/
│   └── prediction-service-python/
├── src/                   # Original v4.0 Python code
├── scripts/               # Original v4.0 scripts
├── database/              # SQL migrations
├── docker-compose.yml     # Microservices orchestration
└── .env                   # API keys (configured ✅)
```

## 🔑 API Keys Configured

- ✅ The Odds API
- ✅ API-Basketball
- ✅ BETSAPI
- ✅ Action Network
- ✅ Kaggle

## 📝 Next Steps

1. **Test Microservices:**
   ```powershell
   docker-compose up -d
   curl http://localhost:8080/health
   ```

2. **Use Original Scripts:**
   ```powershell
   python scripts/predict.py --date today
   ```

3. **Develop Services:**
   - Complete prediction service integration with NBA v4.0 models
   - Implement feature store computation
   - Complete odds ingestion database integration

## 📚 Documentation

- `README.md` - Full documentation
- `docs/` - Technical references
- `setup.ps1` - Setup script (already run)

## 🆘 Troubleshooting

**Docker issues:**
- Ensure Docker Desktop is running
- Check ports 8080, 8081, 8082, 8084, 8085 are available

**Python issues:**
- Ensure Python 3.11+ is installed
- Activate virtual environment: `.\venv\Scripts\Activate.ps1`

**API issues:**
- Verify API keys in `.env` file
- Check API quotas/limits
