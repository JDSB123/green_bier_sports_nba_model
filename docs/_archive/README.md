# Archived Documentation

**Archived:** 2026-01-23

This directory contains deprecated documentation that has been consolidated into the main docs.

## What Happened

The `docs/` folder was consolidated from 24+ files to 9 core documents to eliminate redundancy and confusion.

## Consolidated Files

| Archived File | Merged Into |
|---------------|-------------|
| `DATA_SINGLE_SOURCE_OF_TRUTH.md` | `DATA_GUIDE.md` |
| `DATA_SOURCE_OF_TRUTH.md` | `DATA_GUIDE.md` |
| `DATA_INVENTORY.md` | `DATA_GUIDE.md` |
| `DATA_INGESTION_METHODOLOGY.md` | `DATA_GUIDE.md` |
| `HISTORICAL_DATA.md` | `HISTORICAL_GUIDE.md` |
| `HISTORICAL_DATA_BEST_PRACTICES.md` | `HISTORICAL_GUIDE.md` |
| `HISTORICAL_DATA_STORAGE.md` | `HISTORICAL_GUIDE.md` |
| `ARCHITECTURE_FLOW_AND_ENDPOINTS.md` | `ARCHITECTURE.md` |
| `STACK_FLOW_AND_VERIFICATION.md` | `ARCHITECTURE.md` |
| `AZURE_CONFIG.md` | `AZURE_OPERATIONS.md` |
| `AZURE_CREDENTIALS_GUIDE.md` | `AZURE_OPERATIONS.md` |
| `AZURE_CONTAINER_APP_TROUBLESHOOTING.md` | `AZURE_OPERATIONS.md` |
| `TRACK2_DEPLOYMENT_CONFIG.md` | `AZURE_OPERATIONS.md` |
| `DOCKER_SECRETS.md` | `SECRETS.md` |
| `SECRETS_SETUP_CHECKLIST.md` | `SECRETS.md` |
| `CONSOLIDATED_RUNBOOK.md` | `RUNBOOK.md` |
| `DEV_WORKFLOW.md` | `RUNBOOK.md` |
| `WORKFLOW_AUTOMATION.md` | `RUNBOOK.md` |
| `MODEL_VERIFICATION_GUIDE.md` | `MODELS.md` |
| `FEATURE_ARCHITECTURE_v33.1.0.md` | `MODELS.md` |
| `DOCKER_TROUBLESHOOTING.md` | `RUNBOOK.md` |
| `AZURE_FRONT_DOOR_PREPARATION.md` | Future feature (not implemented) |

## Subdirectories

| Directory | Reason Archived |
|-----------|-----------------|
| `guides/` | Optimization guides (one-time use), merged into core docs |
| `plans/` | Completed planning documents |
| `reports/` | Point-in-time status reports |

## New Structure

See `../` for the new consolidated documentation:

- `ARCHITECTURE.md` - System design, endpoints, data flow
- `DATA_GUIDE.md` - Data sources, ingestion, validation
- `HISTORICAL_GUIDE.md` - Historical data, backtests
- `AZURE_OPERATIONS.md` - Azure config, deployment, troubleshooting
- `SECRETS.md` - Secrets management
- `SECURITY_HARDENING.md` - Security practices
- `MODELS.md` - Model architecture, features, verification
- `RUNBOOK.md` - Operational runbook
- `AUDIT_PREDICTION_SOT.md` - Recent audit (reference)
