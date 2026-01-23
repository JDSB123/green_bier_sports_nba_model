# Documentation Consolidation Plan

**Created:** 2026-01-23  
**Purpose:** Eliminate redundancy, clarify ownership, and reduce cognitive load.

---

## Executive Summary

The `docs/` folder contains **24 markdown files** plus 3 subdirectories (`guides/`, `plans/`, `reports/`). Many documents overlap significantly, causing confusion about which is authoritative. This plan proposes consolidating to **8 core documents** plus archiving the rest.

---

## Current State Analysis

### 🔴 REDUNDANT DOCUMENTS (Same topic, different files)

| Topic | Redundant Files | Problem |
|-------|-----------------|---------|
| **Data Source of Truth** | `DATA_SINGLE_SOURCE_OF_TRUTH.md`, `DATA_SOURCE_OF_TRUTH.md`, `DATA_INVENTORY.md` | 3 files all claim to be THE source of truth for data |
| **Historical Data** | `HISTORICAL_DATA.md`, `HISTORICAL_DATA_BEST_PRACTICES.md`, `HISTORICAL_DATA_STORAGE.md` | 3 files covering the same historical workflow |
| **Secrets/Credentials** | `DOCKER_SECRETS.md`, `SECRETS_SETUP_CHECKLIST.md`, `guides/SECRETS_CONFIGURATION.md` | 3 files about the same secrets setup |
| **Azure Config** | `AZURE_CONFIG.md`, `AZURE_CREDENTIALS_GUIDE.md`, `AZURE_CONTAINER_APP_TROUBLESHOOTING.md` | Could be one Azure operations doc |
| **Deployment** | `TRACK2_DEPLOYMENT_CONFIG.md`, `WORKFLOW_AUTOMATION.md`, `CONSOLIDATED_RUNBOOK.md` | Overlapping deployment instructions |
| **Stack Flow** | `ARCHITECTURE_FLOW_AND_ENDPOINTS.md`, `STACK_FLOW_AND_VERIFICATION.md` | Same architectural diagrams |

### 🟡 OUTDATED OR STALE DOCUMENTS

| File | Issue |
|------|-------|
| `FEATURE_ARCHITECTURE_v33.1.0.md` | Version-specific; should be canonical or removed |
| `AZURE_FRONT_DOOR_PREPARATION.md` | Future/planned feature, not implemented |
| `reports/DEPLOYMENT_SUMMARY_v33.0.20.0.md` | Old version deployment summary |
| `reports/DEPLOYMENT_SUMMARY_v33.0.21.0.md` | Old version deployment summary |
| `reports/FINAL_STATUS_v33.0.21.0.md` | Old version status |
| `reports/PRODUCTION_READINESS_REPORT.md` | Dated Dec 2025, version v33.0.11.0 |
| `reports/TONIGHT_SLATE_20260116.md` | Single-day output, should not be in docs |

### 🟢 USEFUL BUT SCATTERED

| File | Value | Recommendation |
|------|-------|----------------|
| `MODEL_VERIFICATION_GUIDE.md` | Useful | Keep, minor update |
| `SECURITY_HARDENING.md` | Useful | Keep |
| `DATA_INGESTION_METHODOLOGY.md` | Detailed API docs | Keep as reference |
| `AUDIT_PREDICTION_SOT.md` | Recent audit | Keep |
| `DEV_WORKFLOW.md` | Quick start | Merge into README |

---

## Proposed New Structure

### Core Documents (8 files)

| New File | Replaces | Purpose |
|----------|----------|---------|
| `ARCHITECTURE.md` | `ARCHITECTURE_FLOW_AND_ENDPOINTS.md`, `STACK_FLOW_AND_VERIFICATION.md` | System architecture, data flow, endpoints |
| `DATA_GUIDE.md` | `DATA_SINGLE_SOURCE_OF_TRUTH.md`, `DATA_SOURCE_OF_TRUTH.md`, `DATA_INVENTORY.md`, `DATA_INGESTION_METHODOLOGY.md` | All data: sources, ingestion, storage, validation |
| `HISTORICAL_GUIDE.md` | `HISTORICAL_DATA.md`, `HISTORICAL_DATA_BEST_PRACTICES.md`, `HISTORICAL_DATA_STORAGE.md` | Historical data & backtest workflows |
| `AZURE_OPERATIONS.md` | `AZURE_CONFIG.md`, `AZURE_CREDENTIALS_GUIDE.md`, `AZURE_CONTAINER_APP_TROUBLESHOOTING.md`, `TRACK2_DEPLOYMENT_CONFIG.md` | Azure resources, deployment, troubleshooting |
| `SECRETS.md` | `DOCKER_SECRETS.md`, `SECRETS_SETUP_CHECKLIST.md`, `guides/SECRETS_CONFIGURATION.md` | All secrets: local, Docker, Azure Key Vault |
| `SECURITY.md` | `SECURITY_HARDENING.md` | Security practices (keep as-is) |
| `MODELS.md` | `MODEL_VERIFICATION_GUIDE.md`, `FEATURE_ARCHITECTURE_v33.1.0.md` | Model architecture, features, verification |
| `RUNBOOK.md` | `CONSOLIDATED_RUNBOOK.md`, `DEV_WORKFLOW.md`, `WORKFLOW_AUTOMATION.md` | Operations runbook: local → Docker → Azure |

### Subdirectories

| Directory | Action |
|-----------|--------|
| `guides/` | **ARCHIVE** - Content merged into core docs or README |
| `plans/` | **ARCHIVE** - One-time planning docs |
| `reports/` | **ARCHIVE** - Historical reports, not operational |

### Archive Location

Move deprecated docs to `docs/_archive/` with a manifest listing what was archived and why.

---

## File-by-File Disposition

### DELETE (merge content first)

| File | Merge Into |
|------|------------|
| `DATA_SINGLE_SOURCE_OF_TRUTH.md` | `DATA_GUIDE.md` |
| `DATA_SOURCE_OF_TRUTH.md` | `DATA_GUIDE.md` |
| `DATA_INVENTORY.md` | `DATA_GUIDE.md` |
| `HISTORICAL_DATA_BEST_PRACTICES.md` | `HISTORICAL_GUIDE.md` |
| `HISTORICAL_DATA_STORAGE.md` | `HISTORICAL_GUIDE.md` |
| `STACK_FLOW_AND_VERIFICATION.md` | `ARCHITECTURE.md` |
| `AZURE_CREDENTIALS_GUIDE.md` | `AZURE_OPERATIONS.md` |
| `AZURE_CONTAINER_APP_TROUBLESHOOTING.md` | `AZURE_OPERATIONS.md` |
| `TRACK2_DEPLOYMENT_CONFIG.md` | `AZURE_OPERATIONS.md` |
| `DOCKER_SECRETS.md` | `SECRETS.md` |
| `SECRETS_SETUP_CHECKLIST.md` | `SECRETS.md` |
| `guides/SECRETS_CONFIGURATION.md` | `SECRETS.md` |
| `DEV_WORKFLOW.md` | `RUNBOOK.md` + README |
| `WORKFLOW_AUTOMATION.md` | `RUNBOOK.md` |
| `FEATURE_ARCHITECTURE_v33.1.0.md` | `MODELS.md` |
| `DOCKER_TROUBLESHOOTING.md` | `RUNBOOK.md` |

### ARCHIVE (move to `docs/_archive/`)

| File | Reason |
|------|--------|
| `AZURE_FRONT_DOOR_PREPARATION.md` | Future feature, not implemented |
| `guides/MONEYLINE_QUICK_START.md` | Moneyline not in production surface |
| `guides/README_MONEYLINE_OPTIMIZATION.md` | Moneyline not in production surface |
| `guides/README_TOTALS_OPTIMIZATION.md` | Optimization docs (one-time use) |
| `guides/SPREAD_OPTIMIZATION_GUIDE.md` | Optimization docs (one-time use) |
| `guides/TOTALS_OPTIMIZATION_GUIDE.md` | Optimization docs (one-time use) |
| `guides/WEEK1_MONITORING_GUIDE.md` | One-time monitoring setup |
| `plans/moneyline_optimization_plan.md` | Completed plan |
| `reports/*` | All reports are point-in-time snapshots |

### KEEP AS-IS (or minor updates)

| File | Notes |
|------|-------|
| `AUDIT_PREDICTION_SOT.md` | Recent audit, keep for reference |
| `SECURITY_HARDENING.md` → `SECURITY.md` | Rename only |
| `MODEL_VERIFICATION_GUIDE.md` | Merge into `MODELS.md` |

---

## Scripts Audit

### Current: 75 scripts in `/scripts/`

### Categorization

| Category | Count | Scripts |
|----------|-------|---------|
| **Prediction (daily)** | 4 | `predict_unified_*.py` |
| **Data Unified** | 18 | `data_unified_*.py` |
| **Historical** | 9 | `historical_*.py` |
| **Model** | 3 | `model_*.py` |
| **Optimization** | 6 | `optimize_*.py`, `run_spread_optimization.py` |
| **Analysis** | 8 | `analyze_*.py`, `check_*.py`, `inspect_*.py` |
| **Ops/Deploy** | 10 | `manage_secrets.py`, `bump_version.py`, etc. |
| **Azure** | 2 | `upload_training_data_to_azure.py`, `download_training_data_from_azure.py` |
| **Shell/PS** | 6 | `*.sh`, `*.ps1` (in scripts/) |
| **Misc** | 9 | `export_*.py`, `post_to_teams.py`, etc. |

### Scripts to Consider Archiving

| Script | Reason |
|--------|--------|
| `download_kaggle_player_data.py` | One-time data acquisition |
| `fix_training_data_gaps.py` | Migration script, rarely used |
| `inspect_leakage.py` | Debugging utility |
| `compare_versions.py` | Debugging utility |
| `compare_thresholds_api.py` | Testing utility |
| `deploy_option_b.py` | Alternate deployment (unused?) |

---

## Implementation Plan

### Phase 1: Create Archive Structure
```bash
mkdir -p docs/_archive/guides
mkdir -p docs/_archive/plans
mkdir -p docs/_archive/reports
mkdir -p scripts/_archive
```

### Phase 2: Consolidate Core Docs
1. Create `DATA_GUIDE.md` from 4 data docs
2. Create `HISTORICAL_GUIDE.md` from 3 historical docs
3. Create `AZURE_OPERATIONS.md` from 4 Azure docs
4. Create `SECRETS.md` from 3 secrets docs
5. Create `ARCHITECTURE.md` from 2 architecture docs
6. Create `MODELS.md` from 2 model docs
7. Create `RUNBOOK.md` from 3 workflow docs

### Phase 3: Archive Old Docs
Move all deprecated files to `docs/_archive/`

### Phase 4: Update References
- Update README.md to reference new doc structure
- Update `.github/copilot-instructions.md` with new doc paths

### Phase 5: Clean Up Scripts
- Archive unused scripts to `scripts/_archive/`
- Update `scripts/README.md`

---

## Quick Reference: New Structure

```
docs/
├── ARCHITECTURE.md          # System design, endpoints, data flow
├── DATA_GUIDE.md            # Data sources, ingestion, validation
├── HISTORICAL_GUIDE.md      # Historical data, backtests, Azure-only storage
├── AZURE_OPERATIONS.md      # Azure config, deployment, troubleshooting
├── SECRETS.md               # All secrets management
├── SECURITY.md              # Security hardening
├── MODELS.md                # Model architecture, features, verification
├── RUNBOOK.md               # Operational runbook: local → Docker → Azure
├── AUDIT_PREDICTION_SOT.md  # Recent audit (keep for reference)
└── _archive/                # Deprecated docs
    ├── guides/
    ├── plans/
    └── reports/
```

---

## Approval Required

Before executing:
1. Confirm the consolidation targets are correct
2. Confirm which docs should be archived vs deleted
3. Confirm script archive list

**Execute with:** `git add -A && git commit -m "docs: consolidate documentation structure"`
