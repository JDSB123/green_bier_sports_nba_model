#!/bin/bash
# Azure Function App Deployment Script
# Usage: ./azure/deploy.sh <function-app-name> <resource-group>

set -e

FUNCTION_APP_NAME=$1
RESOURCE_GROUP=$2

if [ -z "$FUNCTION_APP_NAME" ] || [ -z "$RESOURCE_GROUP" ]; then
    echo "Usage: $0 <function-app-name> <resource-group>"
    echo "Example: $0 green-bier-sports-nba nba-resources"
    exit 1
fi

echo "🚀 Deploying Azure Function App: $FUNCTION_APP_NAME"

# Change to function app directory
cd "$(dirname "$0")/function_app"

# Install Azure Functions Core Tools if not present
if ! command -v func &> /dev/null; then
    echo "❌ Azure Functions Core Tools not found. Installing..."
    npm install -g azure-functions-core-tools@4 --unsafe-perm true
fi

# Deploy function app
echo "📦 Deploying to Azure..."
func azure functionapp publish $FUNCTION_APP_NAME --python

echo "✅ Deployment complete!"
echo ""
echo "Function endpoints:"
echo "  - Generate Picks: https://${FUNCTION_APP_NAME}.azurewebsites.net/api/generate_picks"
echo "  - Teams Bot: https://${FUNCTION_APP_NAME}.azurewebsites.net/api/teams/bot"
echo "  - Live Tracker: https://${FUNCTION_APP_NAME}.azurewebsites.net/api/live_tracker"