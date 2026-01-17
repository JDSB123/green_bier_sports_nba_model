#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install gh to watch Actions runs."
  exit 0
fi

RUN_ID="$(gh run list --branch "$BRANCH" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)"
if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
  echo "No runs found for branch: $BRANCH"
  exit 0
fi

echo "Watching GitHub Actions run $RUN_ID on branch $BRANCH..."
gh run watch "$RUN_ID" || true
