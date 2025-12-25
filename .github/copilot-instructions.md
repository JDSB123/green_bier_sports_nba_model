<!-- Purpose: concise, actionable repo-specific guidance for AI coding agents -->
# Repo AI assistant notes — NBA Basketball Prediction System (nba-gbsv-model-rg)

**⚠️ SINGLE SOURCE OF TRUTH:** This repo (`JDSB123/green_bier_sports_nba_model`, branch `main`) is **THE** production source code for the NBA picks system. All deployments originate from GitHub main → Docker image → Azure Container App.

Purpose: give an AI coding agent the exact, discoverable facts it needs to make safe, correct edits in this repository.

## **DEPLOYMENT PIPELINE (CRITICAL)**

```
LOCAL WORKSPACE (c:\Users\JB\green-bier-ventures\NBA_main)
    ↓ (make changes, test locally)
    ↓ git commit + git push origin main
GITHUB (JDSB123/green_bier_sports_nba_model:main) ← SOURCE OF TRUTH
    ↓ (manual Docker build)
DOCKER IMAGE BUILD
    ↓ docker build -t nbagbsacr.azurecr.io/nba-gbsv-api:vX.X
    ↓ docker push
AZURE CONTAINER REGISTRY (nbagbsacr)
    ↓ az containerapp update
AZURE CONTAINER APP (nba-gbsv-api) ← PRODUCTION
```

**STEPS TO DEPLOY (DO NOT SKIP):**
1. Commit and push to GitHub: `git push origin main` ✅ **ALWAYS do this first**
2. Build Docker image: `docker build -t nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.2.0 -f Dockerfile.combined .`
3. Push to ACR: `az acr login -n nbagbsacr && docker push nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.2.0`
4. Deploy to Azure: `az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.2.0`
5. Verify: `curl https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io/health`

**⚠️ CRITICAL RULE:** Never let the local workspace drift more than one commit ahead of GitHub. Always push before building Docker images.

---

## **Azure Resource → GitHub Source Code Mapping (nba-gbsv-model-rg)**
| Azure Resource | Type | GitHub Repo | Branch | Current Image |
|----------------|------|-------------|--------|-------|
| `nba-gbsv-api` | Container App | `JDSB123/green_bier_sports_nba_model` | `main` | `nbagbsacr.azurecr.io/nba-gbsv-api:NBA_v33.0.2.0` |
| `nbagbsacr` | Container Registry | — | — | Hosts NBA model images |
| `nbagbs-keyvault` | Key Vault | — | — | Stores: THE-ODDS-API-KEY, API-BASKETBALL-KEY |
| `nba-gbsv-model-env` | Container Apps Environment | — | — | Hosts nba-gbsv-api |

## **Resource Group Organization**
| Resource Group | Purpose | Contains |
|---|---|---|
| **`nba-gbsv-model-rg`** (PRODUCTION) | NBA Picks API production | nba-gbsv-api, Key Vault, Container Registry |
| `greenbier-enterprise-rg` | NCAAF/NCAAM sports models | ncaaf-prediction, ncaam services, databases, storage |
| `chat-jb-t-...` | Chat API | chat-jb-t-app |

**⚠️ CRITICAL:** `nba-gbsv-model-rg` is the ONLY resource group for NBA picks production.

## **Where Secrets & Config Live**
- Secrets are stored in **Azure Key Vault** (`nbagbs-keyvault`), NOT in the repo or `.env` files
- Production deployment reads secrets directly from Key Vault via Container App environment variables
- Local development uses `.env` file (create from `.env.example`) — DO NOT commit
- See `docs/DOCKER_SECRETS.md` for detailed secrets management

## **This Repository Structure (NBA Prediction System)**
- `src/` — Core Python prediction code (ingestion, modeling, serving, tracking, utils)
- `scripts/` — Utility & deployment scripts (see `scripts/README.md`)
- `tests/` — pytest test suite
- `docs/` — Full architecture and operational documentation
- `azure/` — Azure Functions & Teams integration code (supplementary)
- `models/production/` — Trained model files (large, usually git-lfs)
- `data/` — Raw and processed data
- `Dockerfile` / `Dockerfile.backtest` / `Dockerfile.combined` — Container definitions
- `docker-compose.yml` / `docker-compose.backtest.yml` — Service orchestration
- `pyproject.toml` / `requirements.txt` — Python dependencies

## **Development Workflow**
1. Make code changes in local workspace
2. **Test locally** if possible (run scripts, check syntax)
3. **git commit** with clear message
4. **git push origin main** to GitHub ← **This syncs the source of truth**
5. **THEN** proceed to Docker build/push/deploy steps above

## **Build / Run Commands (Local Development)**
- Python environment: `python -m venv .venv && .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`
- Run predictions: `python scripts/predict.py` or `python scripts/run_slate.py`
- Run backtests (Docker): `docker compose -f docker-compose.backtest.yml up backtest-full`
- Run unit tests: `pytest tests -v`
- VS Code tasks available for: Train Models, Generate Predictions, Collect Odds Data, Run Backtest, Run Tests

## **Important Documentation Files**
- `README.md` — Quick start, API usage, architecture overview
- `docs/ARCHITECTURE_FLOW_AND_ENDPOINTS.md` — Detailed stack and API endpoints
- `docs/SINGLE_SOURCE_OF_TRUTH_AUDIT.md` — Data lineage and validation
- `docs/DOCKER_SECRETS.md` — Secret management (Key Vault setup)
- `docs/DOCKER_TROUBLESHOOTING.md` — Common Docker issues
- `scripts/README.md` — Script descriptions and usage

## **Testing & Verification**
- Local: run `pytest tests -v` or use VS Code task "Run Tests"
- Container health: `curl http://localhost:8090/health`
- Production health: `curl https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io/health`
- Logs: `docker compose logs -f nba-v60-api` (local) or `az containerapp logs -n nba-gbsv-api -g nba-gbsv-model-rg` (Azure)

## **Common Pitfalls to Avoid**
- ⚠️ **Drifting from GitHub:** Always `git push origin main` BEFORE building Docker images. Local commits must not drift more than 1 commit ahead.
- ⚠️ **Stale Docker images:** Pushing code to GitHub does NOT auto-deploy. You must manually build/push/update.
- ⚠️ **Secrets in code:** Never hardcode API keys. Use `.env.example` as template; actual keys go in Azure Key Vault.
- ⚠️ **Wrong registry:** Use `nbagbsacr` (not `greenbieracr`) for nba-gbsv-model-rg deployments.
- ⚠️ **Missing credentials:** Docker builds need `secrets/THE_ODDS_API_KEY` and `secrets/API_BASKETBALL_KEY` files. Fetch from Key Vault if missing: `az keyvault secret show --vault-name nbagbs-keyvault --name THE-ODDS-API-KEY --query value -o tsv > secrets/THE_ODDS_API_KEY`

## **Editing Rules for AI Agents**
- **Always push to GitHub first** before any Docker/Azure operations
- Never hardcode secrets in code or `.env` files; use Key Vault only
- When modifying API responses or data structures, update relevant docs
- Preserve the 9-market structure (Q1/1H/FG × spread/total/moneyline)
- Keep the Docker Compose setup intact; modifications should be backward compatible
- Document any new scripts in `scripts/README.md`
