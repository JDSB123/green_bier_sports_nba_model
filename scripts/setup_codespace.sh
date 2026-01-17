#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Install Python 3.11+ and re-run."
  exit 1
fi

mkdir -p secrets

if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
  else
    echo ".env.example not found; skipping .env creation."
  fi
fi

if [ "${USE_SYSTEM_PYTHON:-0}" = "1" ]; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
else
  if [ ! -d .venv ]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  "$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT_DIR/.venv/bin/python" -m pip install -r requirements.txt
fi

chmod +x scripts/*.py 2>/dev/null || true

if command -v git >/dev/null 2>&1; then
  if [ -d "$ROOT_DIR/.githooks" ]; then
    existing_hooks_path="$(git config --get core.hooksPath || true)"
    if [ -z "$existing_hooks_path" ] || [ "$existing_hooks_path" = ".githooks" ]; then
      git config core.hooksPath .githooks
    else
      echo "core.hooksPath already set to $existing_hooks_path; skipping hook setup."
    fi
  fi
fi

cat <<'EOF'
Setup complete.

Next steps:
- Start API: docker compose up -d
- Health check: curl http://localhost:8090/health
- Predictions: curl http://localhost:8090/slate/today
EOF
