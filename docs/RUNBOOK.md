# Operational Runbook

**Last Updated:** 2026-01-23
**Status:** Consolidated from CONSOLIDATED_RUNBOOK, DEV_WORKFLOW, WORKFLOW_AUTOMATION

---

## The One Command

```bash
python scripts/predict_unified_slate.py
```

That's it for daily predictions.

---

## Azure‑First Workflow

Local container runs are intentionally removed to prevent conflicting environments.
Use Azure Container Apps for all production predictions.

---

## Backtest (Scripts)

```bash
# Production backtest (audited canonical dataset)
python scripts/historical_backtest_production.py

# Extended backtest (per-market configuration)
python scripts/historical_backtest_extended.py
```

---

## Azure Deployment

### Quick Deploy

```bash
VERSION=$(cat VERSION)

# Build
docker build --build-arg MODEL_VERSION=$VERSION \
  -t nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION \
  -f Dockerfile.combined .

# Push
az acr login -n nbagbsacr
docker push nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Deploy
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg \
  --image nbagbsacr.azurecr.io/nba-gbsv-api:$VERSION

# Verify
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg \
  --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health"
```

### CI/CD (Automated)

Push to `main` → GitHub Actions → Build → Deploy → Smoke test

Check status: GitHub → Actions tab

---

## Versioning

### Bump Version

```bash
python scripts/bump_version.py NBA_v33.1.9
git add VERSION models/production/model_pack.json
git commit -m "chore: bump version to NBA_v33.1.9"
git push origin main
```

### Version File

```bash
cat VERSION
# NBA_v33.1.8
```

---

## Training Models

```bash
# Train all markets
python scripts/model_train_all.py

# Validate models
python scripts/model_validate.py
```

---

## Common Tasks

| Task | Command |
|------|---------|
| Today's predictions | `python scripts/predict_unified_slate.py` |
| Tomorrow's predictions | `python scripts/predict_unified_slate.py --date tomorrow` |
| Specific team | `python scripts/predict_unified_slate.py --matchup Lakers` |
| Run tests | `pytest tests -v` |
| Validate training data | `python scripts/data_unified_validate_training.py --strict` |
| Train models | `python scripts/model_train_all.py` |
| Backtest | `python scripts/historical_backtest_production.py` |

---

## Troubleshooting

### Version check failed
```bash
python scripts/bump_version.py <VERSION>
git add . && git commit -m "fix: version sync"
```

### Missing secrets
See [SECRETS.md](SECRETS.md)

### Container health fails
```bash
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg
FQDN=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query properties.configuration.ingress.fqdn -o tsv)
curl "https://$FQDN/health"
```

### Azure deployment fails
```bash
az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg
```

---

## Codespaces

One-command setup:
```bash
./scripts/setup_codespace.sh
```

This:
- Creates venv and installs deps
- Syncs Codespaces secrets to `.env` and `secrets/`
- Configures git hooks

Then:
```bash
pytest tests -v
```
