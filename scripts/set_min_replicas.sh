#!/bin/bash

# Set minimum replicas to prevent cold starts in Azure Container App
# Run this script when you have Azure CLI access

set -e

echo "Setting minimum replicas to 1 for nba-gbsv-api..."

az containerapp update \
  -n nba-gbsv-api \
  -g nba-gbsv-model-rg \
  --min-replicas 1 \
  --max-replicas 10

echo "✅ Minimum replicas set to 1. Cold starts prevented."
echo ""
echo "Verify the change:"
echo "az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query '{name:name, minReplicas:scale.minReplicas, maxReplicas:scale.maxReplicas}'"
