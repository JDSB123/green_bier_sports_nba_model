# Scripts Directory

Operational scripts for the NBA prediction system.

Last updated: 2026-01-23

**See also:** [`docs/RUNBOOK.md`](../docs/RUNBOOK.md) for operational workflows.

---

## Production Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CODESPACE / LOCAL                                                       │
│  python scripts/predict_unified_slate.py                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AZURE CONTAINER APP (PRODUCTION)                                        │
│  https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  UnifiedPredictionEngine                                         │    │
│  │  • 4 trained ML models (.joblib files)                          │    │
│  │  • Edge-based filtering thresholds                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LIVE DATA SOURCES (REQUIRED - NOT OPTIONAL)                     │    │
│  │  • The Odds API → spreads, totals, moneylines                   │    │
│  │  • API-Basketball → standings, team stats, scores               │    │
│  │  • Action Network → betting splits (REQUIRED for RLM)           │    │
│  │  • ESPN → standings fallback                                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  DATA STANDARDIZATION (AUTOMATIC)                                        │
│  • Team names → ESPN canonical format                                   │
│  • Dates/times → CST (Central Standard Time)                           │
│  • All sources unified to same schema                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4 Independent Markets

| Period | Market | Model File | Description |
|--------|--------|------------|-------------|
| **1H** | Spread | `1h_spread_model.joblib` | First Half point spread |
| **1H** | Total | `1h_total_model.joblib` | First Half over/under |
| **FG** | Spread | `fg_spread_model.joblib` | Full Game point spread |
| **FG** | Total | `fg_total_model.joblib` | Full Game over/under |

**Note:** `.joblib` files are **serialized ML models** (trained sklearn pipelines), NOT scripts.

---

## Daily Prediction Commands

```bash
# Get today's predictions (calls Azure production API)
python scripts/predict_unified_slate.py

# Tomorrow's slate
python scripts/predict_unified_slate.py --date tomorrow

# Specific date
python scripts/predict_unified_slate.py --date 2026-01-24
```

**Environment Variables:**
- `NBA_API_URL` - Defaults to Azure production. Set to `http://localhost:8090` for local dev.

---

## Website & Teams Integration

- **Website:** https://www.greenbiersportsventures.com (NBA picks page)
- **Teams Webhook:** Configured via `TEAMS_WEBHOOK_URL` in Azure Key Vault
- Both consume the same `/slate/{date}/comprehensive` endpoint

---

## Script Categories

### Prediction (Daily Use)
| Script | Description |
|--------|-------------|
| `predict_unified_slate.py` | **Main entry point** - calls Azure production API |
| `predict_unified_full_game.py` | Standalone predictions (runs locally) |
| `show_executive.py` | Show executive summary |
| `predict_unified_review.py` | Review prediction results |
| `predict_preflight_freshness.py` | Validate data freshness |
| `predict_validate_production_readiness.py` | Validate config, imports, API keys |
| `predict_test_all_api_endpoints.py` | Test all API endpoints |

### Model Training
| Script | Description |
|--------|-------------|
| `model_train_all.py` | Train all 4 market models |
| `model_validate.py` | Validate model files |
| `model_manage.py` | Model file management |

### Data Ingestion (Live Data)
| Script | Description |
|--------|-------------|
| `data_unified_ingest_all.py` | Run full ingestion pipeline |
| `data_unified_fetch_the_odds.py` | Fetch current odds from The Odds API |
| `data_unified_fetch_api_basketball.py` | Fetch game data from API-Basketball |
| `data_unified_fetch_betting_splits.py` | Fetch public betting percentages |
| `data_unified_fetch_injuries.py` | Fetch injury reports |
| `data_unified_fetch_box_scores.py` | Fetch NBA API box scores |

### Training Data Engineering
| Script | Description |
|--------|-------------|
| `data_unified_build_training_complete.py` | Build training data from raw sources |
| `data_unified_feature_complete.py` | Compute model features |
| `data_unified_validate_training.py` | Validate training data quality |
| `data_unified_compute_betting_labels.py` | Compute spread/total labels |
| `fix_training_data_gaps.py` | Fix FG labels, totals, rest days |

### Azure Storage
| Script | Description |
|--------|-------------|
| `upload_training_data_to_azure.py` | Validate then upload to Azure |
| `download_training_data_from_azure.py` | Download canonical training data from Azure |

### Operations
| Script | Description |
|--------|-------------|
| `post_to_teams.py` | Post predictions to Microsoft Teams |
| `prepare_deployment.py` | Prepare deployment package |
| `bump_version.py` | Bump version number |
| `manage_secrets.py` | Docker secrets management |
| `calculate_pick_results.py` | Calculate pick outcomes |
| `update_pick_tracker.py` | Update pick tracking database |
| `ci_sanity_check.py` | CI/CD sanity checks |
| `validate_environment.py` | Validate environment setup |

### Shell Scripts
| Script | Description |
|--------|-------------|
| `setup_codespace.sh` | One-command dev setup (env, deps, hooks) |
| `keepalive.sh` | Keep codespace alive |
| `set_min_replicas.sh` | Set Azure Container App replicas |

---

## Canonical Training Data

The single source of truth for training data:

```
data/processed/training_data.csv
```

| Attribute | Value |
|-----------|-------|
| Date Range | 2023-01-01 to 2026-01-09 |
| Games | 3,969 |
| Columns | 327 |
| FG Labels | 100% coverage |
| 1H Labels | 100% coverage |

### Azure Storage Mirror

```
Storage Account: nbagbsvstrg
Container: nbahistoricaldata
Prefix: training_data/latest/
```

```bash
# Download from Azure
python scripts/download_training_data_from_azure.py --version latest --verify

# Validate locally
python scripts/data_unified_validate_training.py --strict
```

---

## Local Development (Docker)

For local development only. **Production uses Azure Container App.**

```bash
# Start local Docker stack
docker compose up -d

# Check health
curl http://localhost:8090/health

# Run predictions against LOCAL stack (not recommended)
NBA_API_URL=http://localhost:8090 python scripts/predict_unified_slate.py
```

---

## Required Data Sources (ALL REQUIRED)

| Source | Purpose | API Key Env Var | Failure Mode |
|--------|---------|-----------------|--------------|
| **The Odds API** | Live odds, spreads, totals | `THE_ODDS_API_KEY` | ❌ Fatal - no predictions |
| **API-Basketball** | Standings, team stats | `API_BASKETBALL_KEY` | ❌ Fatal - no features |
| **Action Network** | Betting splits (RLM) | (scraping) | ❌ Fatal - no splits |
| **ESPN** | Schedule, standings | (public) | ⚠️ Fallback available |

**CRITICAL:** Betting splits are **REQUIRED** for Reverse Line Movement (RLM) detection.
The system will throw an error if betting splits cannot be fetched.

---

## Archived Scripts

Legacy backtest and historical data scripts have been archived to:
- `scripts/_archive/backtest/` - Backtest and optimization scripts
- `archive/legacy_docker/` - Legacy Docker backtest configuration
- `archive/legacy_src/` - Legacy source modules (github_data.py, historical_guard.py)

These are retained for reference but are not part of the production workflow.
