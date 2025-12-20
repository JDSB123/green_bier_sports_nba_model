# NCAAF Main – Developer Quick Start

This repo hosts the NCAAF project under `NCAAF_main` with:
- Go ingestion services (`ingestion/...`)
- Python ML API (`ml_service/...`)
- Docker Compose files and monitoring stack
- Migrations and schema (`database/`, `ingestion/migrations/`)

## Local Dev

1) Open the folder in VS Code

```
code C:\Users\JB\green-bier-ventures\NCAAF_main
```

2) Start the ML API (auto-creates venv and installs deps)

```
PowerShell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -API
```

Default (no switches) also sets up venv and runs the API:

```
PowerShell -ExecutionPolicy Bypass -File scripts\run_dev.ps1
```

3) Docker Compose (optional)

```
PowerShell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -Compose
PowerShell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -ComposeDown
```

4) Ingestion helpers (Go)

```
PowerShell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -ManualFetch
PowerShell -ExecutionPolicy Bypass -File scripts\run_dev.ps1 -Worker
```

## Notes
- VS Code tasks exist locally under `.vscode/tasks.json` (ignored by .gitignore).
- For team-shared workflows, use the `scripts/run_dev.ps1` entry points.
- Create a feature branch for changes and open a PR to `main`.
