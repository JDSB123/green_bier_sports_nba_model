#!/usr/bin/env bash
set -euo pipefail

# Stop any running keepalive loops started by this workspace
pkill -f "\[codespace-keepalive\]" || true
pkill -f "scripts/keepalive.sh" || true

echo "Stopped keepalive processes (if any)."
