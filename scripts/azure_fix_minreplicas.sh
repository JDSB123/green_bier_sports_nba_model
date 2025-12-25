#!/bin/bash
# Azure Container App - Fix minReplicas Cold Start Issue
#
# This script sets minReplicas=1 to eliminate cold starts (15-30s delay on first request)
#
# Usage:
#   ./scripts/azure_fix_minreplicas.sh
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Access to nba-gbsv-model-rg resource group

set -e

RESOURCE_GROUP="nba-gbsv-model-rg"
APP_NAME="nba-gbsv-api"

echo "=== Azure Container App Cold Start Fix ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "App Name: $APP_NAME"
echo ""

# Check if logged in
if ! az account show > /dev/null 2>&1; then
    echo "ERROR: Not logged in to Azure CLI. Run 'az login' first."
    exit 1
fi

echo "Current configuration:"
az containerapp show -n $APP_NAME -g $RESOURCE_GROUP --query "{minReplicas:properties.template.scale.minReplicas, maxReplicas:properties.template.scale.maxReplicas}" -o table

echo ""
echo "Updating minReplicas to 1 (eliminates cold starts)..."
az containerapp update -n $APP_NAME -g $RESOURCE_GROUP \
    --min-replicas 1 \
    --max-replicas 10

echo ""
echo "Updated configuration:"
az containerapp show -n $APP_NAME -g $RESOURCE_GROUP --query "{minReplicas:properties.template.scale.minReplicas, maxReplicas:properties.template.scale.maxReplicas}" -o table

echo ""
echo "=== Cold Start Fix Complete ==="
echo "The API will now always have at least 1 replica running."
echo "First request latency should be <1s instead of 15-30s."
