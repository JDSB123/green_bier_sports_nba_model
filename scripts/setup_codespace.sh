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

sync_env_var() {
  local key="$1"
  local value="${!key:-}"

  if [ -z "$value" ]; then
    return 0
  fi

  if [ ! -f .env ]; then
    touch .env
  fi

  SYNC_KEY="$key" SYNC_VALUE="$value" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

key = os.environ["SYNC_KEY"]
value = os.environ["SYNC_VALUE"]
path = Path(".env")
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
out = []
found = False

for line in lines:
    if line.startswith(f"{key}="):
        out.append(f"{key}={value}")
        found = True
    else:
        out.append(line)

if not found:
    out.append(f"{key}={value}")

path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
}

write_secret_file() {
  local key="$1"
  local value="${!key:-}"

  if [ -z "$value" ]; then
    return 0
  fi

  local secret_path="secrets/$key"
  printf '%s' "$value" > "$secret_path"
  chmod 600 "$secret_path" 2>/dev/null || true
}

SYNC_KEYS=(
  THE_ODDS_API_KEY
  API_BASKETBALL_KEY
  ACTION_NETWORK_USERNAME
  ACTION_NETWORK_PASSWORD
  SERVICE_API_KEY
  BETSAPI_KEY
  KAGGLE_API_TOKEN
  TEAMS_WEBHOOK_URL
)

for key in "${SYNC_KEYS[@]}"; do
  sync_env_var "$key"
  write_secret_file "$key"
done

require_secret() {
  local key="$1"
  local env_val="${!key:-}"
  local file_path="secrets/$key"

  if [ -n "$env_val" ]; then
    return 0
  fi

  if grep -q "^${key}=" .env 2>/dev/null; then
    return 0
  fi

  if [ -s "$file_path" ]; then
    return 0
  fi

  echo "❌ Missing required secret: $key" >&2
  echo "   Set a Codespaces secret named $key, or populate .env or $file_path." >&2
  return 1
}

# Enforce required secrets early to avoid flaky runs later.
REQUIRED_SECRETS=(
  THE_ODDS_API_KEY
  API_BASKETBALL_KEY
  ACTION_NETWORK_USERNAME
  ACTION_NETWORK_PASSWORD
)

for key in "${REQUIRED_SECRETS[@]}"; do
  require_secret "$key"
done

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

# Run environment validation
echo ""
echo "🔐 Validating environment..."
"$PYTHON_BIN" scripts/validate_environment.py || {
  echo ""
  echo "⚠️  Some secrets are missing. See instructions above."
  echo "   Add secrets at: https://github.com/settings/codespaces"
  echo "   Then restart this Codespace."
}

cat <<'EOF'

Setup complete.

Next steps:
- Validate: python scripts/validate_environment.py
- Start API: docker compose up -d
- Health check: curl http://localhost:8090/health
- Predictions: curl http://localhost:8090/slate/today

Note: Codespaces secrets are synced to .env and secrets/ when present.
EOF
