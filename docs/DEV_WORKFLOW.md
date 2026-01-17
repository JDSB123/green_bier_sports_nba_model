# Development Workflow (NBA Prediction System)

This guide is the repeatable, minimal workflow for local dev, Codespaces, and deployment.

## Codespaces quickstart

1. Create or reopen the Codespace (extensions auto-install).
2. Ensure secrets are available:
   - For containers: create `secrets/THE_ODDS_API_KEY` and `secrets/API_BASKETBALL_KEY` (see `docs/DOCKER_SECRETS.md`).
   - For local runs: copy `.env.example` to `.env` and fill values (never commit).
3. Start the API:
   - `docker compose up -d`
4. Verify:
   - `curl http://localhost:8090/health`

## Local dev quickstart (non-Codespaces)

1. Create venv and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
2. Set `PYTHONPATH` to repo root or use the venv with `python` from repo root.
3. Run scripts:
   - `python scripts/run_slate.py`
   - `python scripts/predict.py`

## Backtesting

- Full backtest (Docker):
  - `docker compose -f docker-compose.backtest.yml up backtest-full`

## Tests

- `pytest tests -v`

## Versioning and release flow

1. Make changes.
2. Run tests if relevant.
3. If releasing or changing model artifacts, bump version:
   - `python scripts/bump_version.py <VERSION>`
4. Commit changes.
5. Push to `main` before any Docker build or deploy:
   - `git push origin main`

Notes:
- CI enforces version consistency across `VERSION`, `models/production/model_pack.json`, and `models/production/feature_importance.json`.
- The deploy workflow runs on push to `main`. Check GitHub Actions if a deploy or validation fails.

## Common troubleshooting

- Version check failed: run `python scripts/bump_version.py <VERSION>` and commit.
- Missing secrets: follow `docs/DOCKER_SECRETS.md`.
- Container health: `curl http://localhost:8090/health`
