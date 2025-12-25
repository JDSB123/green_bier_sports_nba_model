#!/bin/bash

# Fix Azure NBA API - Comprehensive diagnostic and repair script

set -e

echo "🚀 AZURE NBA API DIAGNOSTIC & REPAIR SCRIPT"
echo "=========================================="

# 1. Check Azure CLI authentication
echo ""
echo "1️⃣ CHECKING AZURE AUTHENTICATION..."
if ! az account show > /dev/null 2>&1; then
    echo "❌ Azure CLI not authenticated!"
    echo "Run: az login"
    exit 1
fi

SUBSCRIPTION=$(az account show --query id -o tsv)
echo "✅ Azure CLI authenticated (Subscription: $SUBSCRIPTION)"

# 2. Check container app status
echo ""
echo "2️⃣ CHECKING CONTAINER APP STATUS..."
APP_STATUS=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query provisioningState -o tsv 2>/dev/null || echo "NOT_FOUND")

if [ "$APP_STATUS" = "NOT_FOUND" ]; then
    echo "❌ Container app 'nba-gbsv-api' not found!"
    echo "Run the deployment script: ./deploy_to_azure.sh"
    exit 1
fi

echo "Container app status: $APP_STATUS"

# 3. Check scaling
echo ""
echo "3️⃣ CHECKING SCALING CONFIGURATION..."
MIN_REPLICAS=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query 'scale.minReplicas' -o tsv 2>/dev/null || echo "0")
CURRENT_SCALE=$(az containerapp show -n nba-gbsv-api -g nba-gbsv-model-rg --query 'scale.scaleStatus' -o tsv 2>/dev/null || echo "unknown")

echo "Minimum replicas: $MIN_REPLICAS"
echo "Current scale status: $CURRENT_SCALE"

# 4. Fix scaling if needed
if [ "$MIN_REPLICAS" = "0" ] || [ "$MIN_REPLICAS" = "null" ]; then
    echo ""
    echo "4️⃣ FIXING SCALING (setting min replicas to 1)..."
    az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --min-replicas 1
    echo "✅ Scaling fixed - waiting 30 seconds for deployment..."
    sleep 30
else
    echo ""
    echo "4️⃣ SCALING LOOKS OK - proceeding with tests..."
fi

# 5. Test API health
echo ""
echo "5️⃣ TESTING API HEALTH..."
HEALTH_RESPONSE=$(curl -s --max-time 10 https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io/health)

if [ -z "$HEALTH_RESPONSE" ]; then
    echo "❌ API not responding!"
    echo ""
    echo "🔧 TROUBLESHOOTING STEPS:"
    echo "1. Check container logs: az containerapp logs show -n nba-gbsv-api -g nba-gbsv-model-rg"
    echo "2. Redeploy: ./deploy_to_azure.sh"
    echo "3. Check resource group: az group show -n nba-gbsv-model-rg"
    exit 1
fi

echo "✅ API responding!"
echo "Health response: $HEALTH_RESPONSE"

# 6. Test NBA prediction
echo ""
echo "6️⃣ TESTING NBA PREDICTION..."
PREDICTION_RESPONSE=$(curl -s --max-time 15 -X POST \
  https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io/predict/game \
  -H 'Content-Type: application/json' \
  -d '{"home_team": "Golden State Warriors", "away_team": "Dallas Mavericks", "fg_spread_line": -2.5, "fg_total_line": 220.5, "fg_home_ml": -150, "fg_away_ml": 130}')

if [ -z "$PREDICTION_RESPONSE" ]; then
    echo "❌ Prediction API not working!"
    exit 1
fi

echo "✅ Prediction API working!"
echo "Sample response preview:"
echo "$PREDICTION_RESPONSE" | head -10

# 7. Success message
echo ""
echo "🎉 SUCCESS! YOUR AZURE NBA API IS WORKING!"
echo ""
echo "🌐 API Endpoint: https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io"
echo ""
echo "📊 AVAILABLE ENDPOINTS:"
echo "   GET  /health                    - API status"
echo "   GET  /slate/today              - Today's games with odds"
echo "   POST /predict/game             - Get predictions for specific game"
echo ""
echo "🎯 READY TO GET NBA PICKS!"
echo "Run the curl commands from earlier to get live analysis."
