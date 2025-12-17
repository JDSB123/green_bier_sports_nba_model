# Production Readiness Guide

This document outlines the production-ready features implemented in NBA Prediction System v4.0.

## ✅ Production-Ready Features

### 1. Structured Logging

**What:** JSON-formatted logging throughout the application for easy parsing by log aggregators.

**Where:**
- `src/utils/logging.py` - Centralized logging configuration
- Implemented in:
  - `src/serving/app.py` - API serving
  - `src/ingestion/api_basketball.py` - Data ingestion
  - `src/ingestion/the_odds.py` - Odds API client

**Usage:**
```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("Task completed", extra={"duration": 1.23})
logger.error("Task failed", exc_info=True)
```

**Configuration:**
Set the `LOG_LEVEL` environment variable (default: INFO):
```bash
export LOG_LEVEL=DEBUG
```

### 2. Comprehensive Test Suite

**What:** Expanded test coverage with unit and integration tests.

**Where:**
- `tests/test_config.py` - Configuration and season calculation tests
- `tests/test_logging.py` - Logging utilities tests
- `tests/test_serving.py` - FastAPI serving endpoint tests
- `tests/test_ingestion.py` - Data ingestion tests
- `tests/test_features.py` - Feature engineering tests (existing)
- `tests/test_model_io.py` - Model I/O tests (existing)

**Usage:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_serving.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### 3. Model Versioning & Promotion

**What:** Safe model deployment with explicit versioning and promotion workflow.

**Where:**
- `src/modeling/versioning.py` - Model registry and promotion logic
- `scripts/manage_models.py` - CLI tool for model management
- `src/serving/app.py` - Updated to load production models only

**Usage:**

**Register a model:**
```python
from src.modeling.versioning import ModelRegistry

registry = ModelRegistry("data/processed/models")
registry.register_model(
    name="xgboost_v1",
    version="1.0.0",
    path="xgboost_v1_20251207.joblib",
    metrics={"accuracy": 0.85, "precision": 0.82},
    notes="Initial production model"
)
```

**Promote to production:**
```bash
# Interactive promotion (asks for confirmation)
python scripts/manage_models.py promote --name xgboost_v1 --version 1.0.0

# Auto-confirm
python scripts/manage_models.py promote --name xgboost_v1 --version 1.0.0 --yes
```

**List models:**
```bash
# All models
python scripts/manage_models.py list

# Only production models
python scripts/manage_models.py list --status production

# Show current production model
python scripts/manage_models.py production
```

**Compare models:**
```bash
python scripts/manage_models.py compare --model1 xgboost_v1:1.0.0 --model2 xgboost_v1:1.1.0
```

### 4. Pipeline Orchestration

**What:** Robust pipeline execution with dependency management, retries, and proper error handling.

**Where:**
- `src/pipeline/orchestrator.py` - Orchestration framework
- `scripts/full_pipeline.py` - New orchestrated pipeline

**Features:**
- Task dependency management
- Automatic retries with exponential backoff
- Continue-on-failure configuration per task
- Comprehensive logging of task execution
- Skip conditions for conditional execution

**Usage:**
```bash
# Run the orchestrated pipeline
python scripts/full_pipeline.py

# Skip odds fetching
python scripts/full_pipeline.py --skip-odds

# Skip training
python scripts/full_pipeline.py --skip-train

# Predict for specific date
python scripts/full_pipeline.py --date 2025-12-08
```

**Custom pipelines:**
```python
from src.pipeline import Pipeline

pipeline = Pipeline(name="Custom Pipeline")

pipeline.add_task(
    name="fetch_data",
    func=fetch_data_function,
    max_retries=3,
    continue_on_failure=False,
)

pipeline.add_task(
    name="process_data",
    func=process_data_function,
    dependencies=["fetch_data"],
    max_retries=2,
)

results = await pipeline.run()
```

### 5. Optimized Docker Deployment

**What:** Production-ready Docker configuration with security and size optimizations.

**Where:**
- `Dockerfile` - Multi-stage build with non-root user
- `.dockerignore` - Excludes unnecessary files from image

**Features:**
- Multi-stage build for smaller image size
- Non-root user for security
- Health check endpoint
- Optimized layer caching

**Usage:**
```bash
# Build image
docker build -t nba-prediction:latest .

# Run container
docker run -p 8080:8080 \
  -e THE_ODDS_API_KEY=your_key \
  -e API_BASKETBALL_KEY=your_key \
  -v $(pwd)/data:/app/data \
  nba-prediction:latest

# Check health
curl http://localhost:8080/health
```

## 📋 Pre-Production Checklist

Before deploying to production, ensure:

- [ ] Environment variables are configured:
  - `THE_ODDS_API_KEY`
  - `API_BASKETBALL_KEY`
  - `LOG_LEVEL` (optional, default: INFO)
  - `DATA_PROCESSED_DIR` (optional)

- [ ] A production model has been promoted:
  ```bash
  python scripts/manage_models.py production
  ```

- [ ] Tests are passing:
  ```bash
  pytest
  ```

- [ ] Docker image builds successfully:
  ```bash
  docker build -t nba-prediction:latest .
  ```

- [ ] Health endpoint responds:
  ```bash
  curl http://localhost:8080/health
  ```

## 🚀 Deployment Workflow

### Local Development
```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run pipeline
python scripts/full_pipeline.py

# 4. Train and register model
python scripts/train_models.py
python scripts/manage_models.py promote --name model_name --version 1.0.0

# 5. Start API server
uvicorn src.serving.app:app --reload
```

### Docker Deployment
```bash
# 1. Build image
docker build -t nba-prediction:latest .

# 2. Run container
docker run -d \
  --name nba-api \
  -p 8080:8080 \
  -e THE_ODDS_API_KEY=your_key \
  -e API_BASKETBALL_KEY=your_key \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  nba-prediction:latest

# 3. View logs
docker logs -f nba-api

# 4. Check health
curl http://localhost:8080/health
```

### Kubernetes Deployment (Example)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nba-prediction
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nba-prediction
  template:
    metadata:
      labels:
        app: nba-prediction
    spec:
      containers:
      - name: api
        image: nba-prediction:latest
        ports:
        - containerPort: 8080
        env:
        - name: THE_ODDS_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: the-odds-api-key
        - name: API_BASKETBALL_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: api-basketball-key
        - name: LOG_LEVEL
          value: "INFO"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🔍 Monitoring & Observability

### Logs
All logs are output in JSON format to stdout. Integrate with your logging platform:

- **ELK Stack:** Elasticsearch, Logstash, Kibana
- **Splunk:** Forward logs via Splunk forwarder
- **CloudWatch:** Use AWS CloudWatch agent
- **Datadog:** Use Datadog agent

### Metrics
The `/health` endpoint provides basic health information. Consider adding:

- Prometheus metrics for request rates, latencies
- Custom business metrics (predictions per day, model accuracy)
- Resource utilization metrics (CPU, memory, disk)

### Alerts
Set up alerts for:

- API health check failures
- Model prediction errors (5xx responses)
- Pipeline task failures
- Resource exhaustion (disk space, memory)

## 📝 Best Practices

1. **Model Promotion:** Always test models in a staging environment before promoting to production
2. **Rollback Plan:** Keep previous production models in "archived" status for quick rollback
3. **Monitoring:** Monitor prediction accuracy in production and retrain if performance degrades
4. **Data Quality:** Implement data validation checks in the ingestion pipeline
5. **API Rate Limits:** Monitor API usage to stay within rate limits
6. **Backups:** Regularly backup the model registry and training data

## 🛠️ Troubleshooting

### Model won't load
```bash
# Check registry
python scripts/manage_models.py list --status production

# If no production model, promote one
python scripts/manage_models.py promote --name model_name --version version
```

### Pipeline fails
```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python scripts/full_pipeline.py

# Check individual task logs
```

### API returns 503
- Model not loaded - check logs for model loading errors
- Ensure production model exists and is accessible

### Tests failing
```bash
# Install test dependencies
pip install -r requirements.txt

# Run specific test with verbose output
pytest tests/test_serving.py -v

# Check for missing dependencies
pip list
```

## 📚 Additional Resources

- **API Documentation:** Start server and visit `http://localhost:8080/docs`
- **Model Artifacts:** `data/processed/models/`
- **Logs:** Check stdout/stderr or configured log destination
- **Code Documentation:** See docstrings in source files

