#!/bin/bash

# NBA Prediction System - Manual Azure Deployment
# Run this script on a machine with Azure CLI installed and authenticated

set -e

echo "🚀 NBA Prediction System - Manual Azure Deployment"
echo "=================================================="

# Configuration
ACR_NAME="nbagbsacr"
RESOURCE_GROUP="nba-gbsv-model-rg"
CONTAINER_APP_NAME="nba-gbsv-api"
IMAGE_NAME="nba-gbsv-api"
TAG="latest"

echo "📋 Configuration:"
echo "   • Azure Container Registry: $ACR_NAME"
echo "   • Resource Group: $RESOURCE_GROUP"
echo "   • Container App: $CONTAINER_APP_NAME"
echo "   • Image: $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG"
echo ""

# Check Azure CLI authentication
echo "🔍 Checking Azure CLI authentication..."
if ! az account show > /dev/null 2>&1; then
    echo "❌ Azure CLI not authenticated. Please run:"
    echo "   az login"
    exit 1
fi

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "✅ Azure CLI authenticated (Subscription: $SUBSCRIPTION_ID)"

# Verify resource group exists
echo ""
echo "🔍 Verifying Azure resources..."
if ! az group show --name "$RESOURCE_GROUP" > /dev/null 2>&1; then
    echo "❌ Resource group '$RESOURCE_GROUP' not found"
    exit 1
fi
echo "✅ Resource group '$RESOURCE_GROUP' exists"

# Verify ACR exists and login
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" > /dev/null 2>&1; then
    echo "❌ Azure Container Registry '$ACR_NAME' not found"
    exit 1
fi
echo "✅ Azure Container Registry '$ACR_NAME' exists"

echo ""
echo "🐳 Logging into Azure Container Registry..."
az acr login --name "$ACR_NAME"
echo "✅ ACR login successful"

# Build Docker image
echo ""
echo "🏗️ Building Docker image..."
docker build -t "$IMAGE_NAME:$TAG" .
echo "✅ Docker build completed"

# Tag and push image
FULL_IMAGE_NAME="$ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG"
echo ""
echo "🏷️ Tagging and pushing image..."
docker tag "$IMAGE_NAME:$TAG" "$FULL_IMAGE_NAME"
docker push "$FULL_IMAGE_NAME"
echo "✅ Image pushed to ACR: $FULL_IMAGE_NAME"

# Deploy to Azure Container Apps
echo ""
echo "🚀 Deploying to Azure Container Apps..."
az containerapp update \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --image "$FULL_IMAGE_NAME" \
  --set-env-vars "NBA_MODEL_VERSION=NBA_v33.0.2.0"

echo "✅ Deployment initiated"

# Wait a moment for deployment to start
echo ""
echo "⏳ Waiting for deployment to complete..."
sleep 10

# Check deployment status
echo ""
echo "📊 Checking deployment status..."
DEPLOYMENT_STATUS=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.provisioningState" \
  -o tsv)

echo "📊 Deployment status: $DEPLOYMENT_STATUS"

# Test the deployment
echo ""
echo "🧪 Testing deployed application..."
APP_URL=$(az containerapp show \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" \
  -o tsv)

if [ -n "$APP_URL" ]; then
    echo "🌐 Application URL: https://$APP_URL"

    # Test health endpoint
    if curl -s "https://$APP_URL/health" > /dev/null 2>&1; then
        echo "✅ Health check passed!"
        echo ""
        echo "🎉 DEPLOYMENT SUCCESSFUL!"
        echo "   • Application is running at: https://$APP_URL"
        echo "   • Health endpoint: https://$APP_URL/health"
        echo "   • API documentation: https://$APP_URL/docs"
    else
        echo "⚠️ Health check failed - deployment may still be in progress"
        echo "   Monitor the application at: https://$APP_URL/health"
    fi
else
    echo "❌ Could not retrieve application URL"
fi

echo ""
echo "📋 Deployment Summary:"
echo "   • Image: $FULL_IMAGE_NAME"
echo "   • Container App: $CONTAINER_APP_NAME"
echo "   • Resource Group: $RESOURCE_GROUP"
echo "   • Status: $DEPLOYMENT_STATUS"
