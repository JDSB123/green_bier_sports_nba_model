#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASHRC="$HOME/.bashrc"
VENVSOURCE="source ${ROOT_DIR}/.venv/bin/activate"

# Start keepalive in background.
nohup bash -lc "${ROOT_DIR}/scripts/keepalive.sh" >/tmp/codespace-keepalive.log 2>&1 &

# Ensure venv auto-activates for new shells (idempotent).
if [ ! -f "$BASHRC" ]; then
  touch "$BASHRC"
fi
if ! grep -Fq "$VENVSOURCE" "$BASHRC"; then
  {
    echo "# Auto-activate project venv"
    echo "$VENVSOURCE"
  } >> "$BASHRC"
fi
