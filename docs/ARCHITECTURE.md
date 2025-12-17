# NBA Prediction System v4.0 - Architecture Overview

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  The Odds API    │  │  API-Basketball  │  │  Injury Data     │  │
│  │  (odds, lines)   │  │  (games, stats)  │  │  (player status) │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                     │              │
│           v                     v                     v              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/ingestion/                                               │  │
│  │  - the_odds.py       (async HTTP client)                      │  │
│  │  - api_basketball.py (async HTTP client with retry)           │  │
│  │  - injuries.py       (injury data fetcher)                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│                              v                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  data/raw/                                                    │  │
│  │  - the_odds/odds_TIMESTAMP.json                               │  │
│  │  - api_basketball/games_TIMESTAMP.json                        │  │
│  │  - api_basketball/statistics_TIMESTAMP.json                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         PROCESSING LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  scripts/process_odds_data.py                                 │  │
│  │  - Parse odds data                                            │  │
│  │  - Extract first-half lines                                   │  │
│  │  │  - Calculate betting splits                                   │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  scripts/build_rich_features.py                               │  │
│  │  - Merge games + stats + h2h + odds                           │  │
│  │  - Generate rolling averages (3/5/10 game windows)            │  │
│  │  - Create advanced metrics (pace, efficiency, rest days)      │  │
│  │  - Handle team name mapping                                   │  │
│  │                                                               │  │
│  │  scripts/build_training_dataset.py                            │  │
│  │  - Link normalized odds (`odds_the_odds.csv`) with            │  │
│  │    API-Basketball outcomes (`game_outcomes.csv`)              │  │
│  │  - Produce `training_data.csv` for model training             │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  data/processed/                                              │  │
│  │  - training_data.csv  (features + labels)                     │  │
│  │  - historical_games.csv                                       │  │
│  │  - injuries.csv                                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/modeling/features.py                                     │  │
│  │                                                               │  │
│  │  Core Features:                                               │  │
│  │  - Team statistics (PPG, FG%, 3P%, rebounds, assists)        │  │
│  │  - Rolling averages (3/5/10 game windows)                    │  │
│  │  - Head-to-head history                                      │  │
│  │  - Home/away splits                                          │  │
│  │  - Rest days between games                                   │  │
│  │  - Pace and efficiency ratings                               │  │
│  │  - Odds-derived features (implied probability, EV)           │  │
│  │  - Betting market consensus                                  │  │
│  │                                                               │  │
│  │  Feature Groups (defined in feature_config.py):              │  │
│  │  - CORE: Essential game/team features                        │  │
│  │  - ROLLING: Time-based aggregations                          │  │
│  │  - ADVANCED: Derived metrics                                 │  │
│  │  - ODDS: Market-based features                               │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  scripts/train_models.py                                      │  │
│  │                                                               │  │
│  │  Base Models:                                                 │  │
│  │  - Logistic Regression (baseline)                            │  │
│  │  - Random Forest                                             │  │
│  │  - XGBoost (optional, if installed)                          │  │
│  │  - LightGBM (optional, if installed)                         │  │
│  │                                                               │  │
│  │  Training Process:                                            │  │
│  │  1. Load training_data.csv                                   │  │
│  │  2. Time-based train/test split                              │  │
│  │  3. Feature selection & preprocessing                        │  │
│  │  4. Train each model                                         │  │
│  │  5. Evaluate on test set                                     │  │
│  │  6. Save model artifacts                                     │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  scripts/train_ensemble_models.py                             │  │
│  │  - Combine base model predictions                            │  │
│  │  - Meta-learner stacking                                     │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/modeling/io.py                                           │  │
│  │  - save_model() → .joblib with metadata                      │  │
│  │  - load_model() → restore pipeline                           │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  data/processed/models/                                       │  │
│  │  - model_TIMESTAMP.joblib                                     │  │
│  │  - registry.json (all models)                                │  │
│  │  - production.json (current production model pointer)        │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL VERSIONING LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/modeling/versioning.py                                   │  │
│  │                                                               │  │
│  │  ModelRegistry:                                               │  │
│  │  - register_model()      → Add new model to registry         │  │
│  │  - promote_to_production() → Set as production model         │  │
│  │  - get_production_model() → Retrieve current prod model      │  │
│  │  - list_models()         → Query registry                    │  │
│  │  - compare_models()      → Compare metrics                   │  │
│  │                                                               │  │
│  │  Model Lifecycle:                                             │  │
│  │  candidate → production → archived                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  scripts/manage_models.py (CLI)                               │  │
│  │  - list                                                       │  │
│  │  - promote --name X --version Y                              │  │
│  │  - production                                                │  │
│  │  - compare --model1 X:Y --model2 A:B                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         SERVING LAYER (API)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/serving/app.py (FastAPI)                                 │  │
│  │                                                               │  │
│  │  Endpoints:                                                   │  │
│  │  - GET  /health       → Model status check                   │  │
│  │  - POST /predict      → Generate prediction                  │  │
│  │                                                               │  │
│  │  Startup:                                                     │  │
│  │  1. Load production model (from production.json)             │  │
│  │  2. Initialize pipeline                                      │  │
│  │  3. Ready to serve requests                                  │  │
│  │                                                               │  │
│  │  Request Flow:                                                │  │
│  │  Client → POST /predict → Pipeline → predict_proba()         │  │
│  │        ← Probabilities ←                                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Docker Container                                             │  │
│  │  - Multi-stage build                                         │  │
│  │  - Non-root user                                             │  │
│  │  - Health check                                              │  │
│  │  - Port 8080                                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/pipeline/orchestrator.py                                 │  │
│  │                                                               │  │
│  │  Pipeline Framework:                                          │  │
│  │  - Task dependency management                                │  │
│  │  - Automatic retries with exponential backoff                │  │
│  │  - Continue-on-failure configuration                         │  │
│  │  - Skip conditions                                           │  │
│  │  - Comprehensive logging                                     │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                           │                                          │
│                           v                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  scripts/full_pipeline.py                                  │  │
│  │                                                               │  │
│  │  Task DAG:                                                    │  │
│  │                                                               │  │
│  │  fetch_odds (optional)                                        │  │
│  │      │                                                        │  │
│  │      ├───> fetch_injuries                                     │  │
│  │      │                                                        │  │
│  │      └───> process_odds                                       │  │
│  │                │                                              │  │
│  │                v                                              │  │
│  │      build_training_dataset                                   │  │
│  │                │                                              │  │
│  │                v                                              │  │
│  │         train_models                                          │  │
│  │                │                                              │  │
│  │                v                                              │  │
│  │        train_ensemble                                         │  │
│  │                │                                              │  │
│  │                v                                              │  │
│  │     generate_predictions                                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       OBSERVABILITY LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  src/utils/logging.py                                         │  │
│  │                                                               │  │
│  │  Features:                                                    │  │
│  │  - JSON-formatted logs                                       │  │
│  │  - Configurable log levels (LOG_LEVEL env var)               │  │
│  │  - Exception tracking                                        │  │
│  │  - Structured metadata                                       │  │
│  │                                                               │  │
│  │  Integration Points:                                          │  │
│  │  - All ingestion modules                                     │  │
│  │  - Serving API                                               │  │
│  │  - Pipeline orchestrator                                     │  │
│  │  - Model versioning                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 Technology Stack

### Core Languages & Frameworks
- **Python 3.11+** - Primary language
- **FastAPI** - API serving framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **joblib** - Model serialization

### Machine Learning
- **scikit-learn** - Core ML library (models, pipelines, preprocessing)
- **XGBoost** - Gradient boosting (optional)
- **LightGBM** - Gradient boosting (optional)

### HTTP & APIs
- **httpx** - Async HTTP client
- **requests** - HTTP client (legacy)
- **tenacity** - Retry logic

### Data Ingestion
- **The Odds API** - Betting odds and lines
- **API-Basketball** - NBA game data and statistics
- **rapidfuzz** - Team name matching

### Development & Testing
- **pytest** - Testing framework
- **black** - Code formatting
- **mypy** - Type checking (configured)

### Deployment
- **Docker** - Containerization
- **python-dotenv** - Environment management

### Utilities
- **matplotlib** - Visualization
- **tabulate** - CLI table formatting
- **pyyaml** - Configuration files

## 🔄 Data Flow

### Training Pipeline
```
Raw Data → Processing → Features → Training → Model → Registry → Production
```

**Detailed:**
1. **Ingestion**: APIs → `data/raw/`
2. **Processing**: Raw data → merged datasets → `data/processed/`
3. **Feature Engineering**: Dataset builder (`build_training_dataset.py`) → `training_data.csv`
4. **Training**: Features → ML models → `.joblib` files
5. **Registration**: Model → registry with metadata
6. **Promotion**: Candidate → production (manual approval)

### Prediction Pipeline
```
New Game → Features → Production Model → Predictions
```

**Detailed:**
1. **Fetch**: Get today's games and odds
2. **Build Features**: Merge with historical data, calculate rolling stats
3. **Load Model**: Get production model from registry
4. **Predict**: Generate probabilities for each game
5. **Output**: Predictions with EV calculations
6. **Review**: Run `review_predictions.py` to grade previous slate (full-game + 1H ROI)

## 🗂️ Directory Structure

```
nba_v4.0/
├── src/                          # Source code
│   ├── config.py                 # Configuration & settings
│   ├── ingestion/                # Data collection
│   │   ├── the_odds.py          # Odds API client
│   │   ├── api_basketball.py    # Basketball API client
│   │   ├── injuries.py          # Injury data
│   │   └── team_mapping.json    # Team name standardization
│   ├── modeling/                 # ML components
│   │   ├── features.py          # Feature engineering
│   │   ├── feature_config.py    # Feature definitions
│   │   ├── models.py            # Model definitions
│   │   ├── dataset.py           # Dataset utilities
│   │   ├── io.py                # Model save/load
│   │   ├── model_tracker.py     # Training tracking
│   │   └── versioning.py        # Model registry (NEW)
│   ├── pipeline/                 # Orchestration (NEW)
│   │   ├── orchestrator.py      # Pipeline framework
│   │   └── __init__.py
│   ├── serving/                  # API serving
│   │   └── app.py               # FastAPI application
│   └── utils/                    # Utilities
│       ├── logging.py           # Structured logging (NEW)
│       └── team_names.py        # Team name utilities
│
├── scripts/                      # Executable scripts
│   ├── collect_*.py             # Data collection scripts
│   ├── build_rich_features.py   # Feature generation
│   ├── build_training_dataset.py
│   ├── generate_training_data.py
│   ├── train_models.py          # Model training
│   ├── train_ensemble_models.py
│   ├── predict.py               # Generate predictions
│   ├── analyze_todays_slate.py  # Daily analysis
│   ├── archive_processed_cache.py
│   ├── review_predictions.py    # Grade picks vs results
│   ├── full_pipeline.py         # Original pipeline
│   ├── full_pipeline_v2.py      # Orchestrated pipeline (NEW)
│   ├── manage_models.py         # Model management CLI (NEW)
│   ├── backtest*.py             # Backtesting tools
│   └── validate_leakage.py      # Data validation
│
├── tests/                        # Test suite
│   ├── test_config.py           # Config tests (NEW)
│   ├── test_logging.py          # Logging tests (NEW)
│   ├── test_serving.py          # API tests (NEW)
│   ├── test_ingestion.py        # Ingestion tests (NEW)
│   ├── test_features.py         # Feature tests
│   └── test_model_io.py         # Model I/O tests
│
├── data/                         # Data storage
│   ├── raw/                     # Raw API responses
│   │   ├── the_odds/
│   │   └── api_basketball/
│   └── processed/               # Processed data
│       ├── training_data.csv
│       ├── historical_games.csv
│       ├── models/              # Model artifacts
│       │   ├── *.joblib
│       │   ├── registry.json    # Model registry (NEW)
│       │   └── production.json  # Production pointer (NEW)
│       └── cache/               # Feature cache
│
├── docs/                         # Documentation
│   └── archive/                 # Archived docs
│
├── .dockerignore                 # Docker build exclusions (NEW)
├── Dockerfile                    # Container definition (UPDATED)
├── requirements.txt              # Python dependencies (UPDATED)
├── pyproject.toml               # Project metadata
├── setup.py                     # Package setup
├── pytest.ini                   # Test configuration
├── ARCHITECTURE.md              # This file (NEW)
├── PRODUCTION_READY.md          # Deployment guide (NEW)
└── NBA_v4.0_MODEL.md           # Model documentation
```

## 🔑 Key Design Principles

### 1. **Separation of Concerns**
- Ingestion, processing, modeling, and serving are decoupled
- Each module has a single, well-defined responsibility

### 2. **Configuration-Driven**
- API keys and settings in environment variables
- Feature configuration in dedicated files
- No hardcoded values

### 3. **Async-First**
- HTTP clients use asyncio for performance
- Pipeline can run tasks concurrently
- Non-blocking I/O operations

### 4. **Production-Ready**
- Structured logging for observability
- Health checks for monitoring
- Model versioning for safety
- Comprehensive testing
- Containerized deployment

### 5. **Fail-Safe**
- Retry logic on API calls
- Continue-on-failure for non-critical tasks
- Graceful degradation
- Explicit error handling

### 6. **Data Integrity**
- Timestamped raw data files (immutable)
- Time-aware train/test splits
- Leakage validation
- Feature caching

## 🎯 Execution Modes

### Development Mode
```bash
# Run individual components
python scripts/collect_the_odds.py
python scripts/build_rich_features.py
python scripts/train_models.py
python scripts/predict.py
```

### Production Pipeline (Orchestrated)
```bash
# Full pipeline with orchestration
python scripts/full_pipeline.py
```

### API Serving
```bash
# Local development
uvicorn src.serving.app:app --reload

# Production (Docker)
docker run -p 8080:8080 nba-prediction:latest
```

### Model Management
```bash
# Register and promote models
python scripts/manage_models.py list
python scripts/manage_models.py promote --name xgboost --version 1.0.0
```

## 📊 Model Performance Tracking

```
Training → Evaluation → Registration → Comparison → Promotion → Serving
```

Each step logs:
- Training metrics (accuracy, precision, recall, ROI)
- Test set performance
- Feature importance
- Model metadata (timestamp, hyperparameters)

Registry maintains full history for:
- Model comparison
- Rollback capability
- Performance auditing

## 🔐 Security Considerations

1. **API Keys**: Stored in environment variables, never committed
2. **Docker**: Non-root user, minimal attack surface
3. **Dependencies**: Pinned versions in requirements.txt
4. **Input Validation**: Pydantic models validate all API inputs
5. **Error Handling**: No sensitive data in error messages

## 📈 Scalability

- **Horizontal**: Multiple API containers behind load balancer
- **Vertical**: Async I/O allows high concurrency per instance
- **Caching**: Feature cache reduces computation
- **Model Loading**: Lazy loading, single model per container

## 🎓 Best Practices Implemented

✅ Type hints throughout codebase  
✅ Docstrings on all public functions  
✅ Comprehensive error handling  
✅ Structured logging  
✅ Unit and integration tests  
✅ CI/CD ready (test suite)  
✅ Containerized deployment  
✅ Environment-based configuration  
✅ API documentation (FastAPI auto-docs)  
✅ Version control (Git)  
✅ Model versioning  
✅ Monitoring endpoints  

---

**This architecture is production-ready and follows industry best practices for ML systems.**

