<!-- Purpose: concise, actionable repo-specific guidance for AI coding agents -->
# Repo AI assistant notes — green-bier-sport-ventures-dashboard-v3.3

Purpose: give an AI coding agent the exact, discoverable facts it needs to make safe, correct edits in this repository.

- **Big picture / major components**
  - `Chat_JBt/` — Node/TypeScript, provider-agnostic chat API. Local dev: `npm install` + `npm run dev` (reads `config/chatjbt.env`). Deploys via `deploy-azure.ps1` and uses Azure Key Vault for secrets. See `Chat_JBt/README.md` and `Chat_JBt/package.json`.
  - `dashboard_v3.3/` — Frontend + backend pieces for the dashboard. The serverless API is under `dashboard_v3.3/api` (Azure Functions). VS Code tasks exist to run the Functions host (`func: host start`) — use `npm install` in that folder first.
  - `nfl_v8.0_BETA/` — NFL prediction model. Python. See `nfl_v8.0_BETA/README.md`. GitHub: `JDSB123/nfl_v8.0_BETA` (branch: `master`).
  - `ncaaf_v5.0_BETA/` — NCAAF prediction model. Python. GitHub: `JDSB123/ncaaf_v5.0_BETA` (branch: `master`).
  - `nba_v5.1_model_FINAL/` — **NBA prediction model v6.0**. Python. GitHub: `JDSB123/green_bier_sports_nba_v5.1_model` (branch: `master`, NOT main). 7 markets: Q1/1H/FG spread, total, moneyline.
  - Small utilities / trackers at repo root (`live_picks_tracker.py`, `pick_tracker_beta_v1.0/`) — mainly Python scripts, keep changes minimal and test locally.

- **Azure Resource → GitHub Source Code Mapping (NBAGBSVMODEL)**
  | Azure Resource | Type | GitHub Repo | Branch | Image |
  |----------------|------|-------------|--------|-------|
  | `nba-picks-api` | Container App | `JDSB123/green_bier_sports_nba_v5.1_model` | `master` | `greenbieracr.azurecr.io/nba-model:v6.0` |
  | `ncaam-prediction` | Container App | `JDSB123/green_bier_sports_ncaam_model` | `master` | `greenbieracr.azurecr.io/ncaam-prediction:latest` |
  | `ncaam-postgres` | Container App | (standard postgres image) | — | `postgres:15` |
  | `ncaam-redis` | Container App | (standard redis image) | — | `redis:7` |
  | `greenbier-keyvault` | Key Vault | — | — | Stores: THE-ODDS-API-KEY, API-BASKETBALL-KEY |
  | `greenbieracr` | Container Registry | — | — | Hosts all model images |

- **Where secrets & config live**
  - `Chat_JBt/config/chatjbt.env` — single source of keys for the chat API when running locally. When deployed, secrets are stored in Azure Key Vault (see `Chat_JBt/README.md`).
  - `config_global/` and various `config/` folders — different components read env files from their local `config/` directory. Do not hardcode secrets.

- **Build / dev / run commands (concrete examples)**
  - Chat API (local):
    - `cd Chat_JBt && npm install && npm run dev` (uses `tsx watch src/index.ts`).
    - Docker: `docker build -t chatjbt:latest .` then `docker run -p 8080:8080 --env-file config/chatjbt.env chatjbt:latest`.
  - Dashboard API (Azure Functions):
    - Use the provided VS Code task: `func: host start` (task defined to run from `dashboard_v3.3\api`). Run `npm install` in that folder first.
    - Install step in workspace tasks: `npm install (functions)` then `func: host start`.
  - Frontend (Chat_JBt web or dashboard frontend): many frontends are Vite/React or static (`web/index.html`). Follow the `package.json` scripts inside those folders.
  - Python (models): `python -m venv .venv && .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt` then run scripts like `python run_model.py`.

- **Project-specific patterns & conventions**
  - Use local `config/*.env` files for development only. Production secrets are injected into Azure Key Vault and not committed.
  - PowerShell deployment scripts are the canonical deploy path (see `Chat_JBt/deploy-azure.ps1` and `Chat_JBt/Dockerfile`). Prefer following those for Azure infra changes.
  - Many Node services are ESM-style (`"type":"module"` in `package.json`). Use modern imports and `tsx`/`ts-node` for dev runs.
  - Frontend assets sometimes live as static files (`Chat_JBt/web/index.html`) — code changes may not need a build step.

- **Important files to reference when editing**
  - `Chat_JBt/README.md` — authoritative local dev + deploy steps for the chat API.
  - `Chat_JBt/package.json` — scripts and dependencies (dev uses `tsx watch`).
  - `Chat_JBt/config/chatjbt.env` — expected env keys and examples (do not create secrets in PRs).
  - `dashboard_v3.3/api/` — Azure Functions code and `package.json` for that subproject. Use the existing VS Code tasks to run locally.
  - `nfl_v7.0/requirements.txt` and `pyproject.toml` — Python dependency sources for model code.
  - `deploy-azure.ps1` and `deploy.ps1` (various folders) — follow these for infra or container updates.

- **Cross-component interactions**
  - Chat API <-> Providers: the chat API proxies to third-party providers; keys are rotated via comma-separated env values. See `Chat_JBt/README.md` for provider selection rules (`provider: "auto"`).
  - Dashboard frontend <-> dashboard API: static frontend talks to the Azure Functions backend; confirm endpoints in `dashboard_v3.3/config.local.js` or `config.production.js` before changing CORS or endpoints.
  - Shared data: pick uploads and cached data are managed in `dashboard_v3.3/mysportsbook_uploads` and `dashboard_v3.3/data_cache` — preserve file formats when modifying ingestion code.

- **Testing & verification guidance (practical)**
  - For Node/TS changes: run the local dev script in the target subfolder (e.g., `Chat_JBt/npm run dev`) and smoke-test the HTTP endpoints (`/health`, `/v1/chat/completions`).
  - For Azure Functions: prefer running `func host start` via the workspace task so the runtime matches expectations.
  - For Python/model changes: run small sample scripts in `nfl_v8.0_BETA/` (e.g., `run_model.py`) and keep results deterministic by using pinned `requirements.txt`.

- **Deploying Model Changes to Azure Container Apps**
  1. **Build locally:** `cd <model_folder> && docker build -t greenbieracr.azurecr.io/<image>:<tag> .`
  2. **Push to ACR:** `az acr login -n greenbieracr && docker push greenbieracr.azurecr.io/<image>:<tag>`
  3. **Update Container App:** `az containerapp update -n <app-name> -g NBAGBSVMODEL --image greenbieracr.azurecr.io/<image>:<tag>`
  4. **Restart if needed:** `az containerapp revision restart -n <app-name> -g NBAGBSVMODEL --revision <revision-name>`
  5. **Verify:** `curl -s https://<app-fqdn>/health`

- **Common Pitfalls to Avoid**
  - **Empty workspace folders:** If a model folder (e.g., `nba_v5.1_model_FINAL/`) is empty, clone from GitHub: `git clone https://github.com/JDSB123/<repo>.git <folder> --branch master`
  - **Wrong branch:** Most model repos use `master` as the active branch, NOT `main`. Always check.
  - **Stale images:** After pushing code to GitHub, you MUST rebuild and push the Docker image to ACR. GitHub pushes do NOT auto-deploy.
  - **Secrets for Docker builds:** NBA model requires `secrets/THE_ODDS_API_KEY` and `secrets/API_BASKETBALL_KEY` files before building. Get from Key Vault: `az keyvault secret show --vault-name greenbier-keyvault --name <secret-name> --query value -o tsv`

- **Editing rules for AI agents**
  - Never add or expose secrets in code or committed env files. If a change requires new secrets, add clear instructions in the PR and update the appropriate `deploy-*.ps1` to accept them via Key Vault.
  - When changing public API shapes (e.g., `POST /v1/chat/completions`), update `Chat_JBt/README.md` and add a small integration smoke test.
  - Preserve existing ESM/CommonJS style per-package. Match `package.json`'s `type` and local build scripts.

If anything here is unclear or you want the file to include more examples (test commands, CI notes, or common PR pitfalls), tell me which area to expand. — copilot
