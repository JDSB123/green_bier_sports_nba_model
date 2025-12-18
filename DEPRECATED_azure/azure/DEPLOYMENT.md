# Azure Function App Deployment Guide

## Overview

This guide walks you through deploying the NBA prediction system to Azure Functions with Microsoft Teams integration.

## Prerequisites

1. **Azure Account**: Ensure you have an active Azure subscription
2. **Azure CLI**: Install and configure Azure CLI (`az login`)
3. **Azure Functions Core Tools**: Install for local development
   ```powershell
   npm install -g azure-functions-core-tools@4 --unsafe-perm true
   ```
4. **Python 3.11**: Required for local testing

## Architecture

The Azure Function App includes:

1. **generate_picks**: HTTP-triggered function to generate NBA picks
2. **teams_bot**: Microsoft Teams bot endpoint for command-based pick generation
3. **live_tracker**: HTML dashboard for live pick tracking

## Deployment Steps

### 1. Create Azure Resources

Run the setup script to create all required Azure resources:

```powershell
.\azure\setup_function_app.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources" `
    -Location "eastus"
```

This will create:
- Resource Group
- Storage Account
- App Service Plan (Consumption/Linux)
- Function App
- Application Insights

### 2. Configure Application Settings

Set required API keys and configuration:

```powershell
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings `
        THE_ODDS_API_KEY="your-odds-api-key" `
        API_BASKETBALL_KEY="your-basketball-api-key" `
        ACTION_NETWORK_USERNAME="your-username" `
        ACTION_NETWORK_PASSWORD="your-password"
```

**Alternative: Use Azure Key Vault (Recommended for Production)**

```powershell
# Create Key Vault
az keyvault create --name "nba-keyvault" --resource-group "nba-resources"

# Store secrets
az keyvault secret set --vault-name "nba-keyvault" --name "TheOddsApiKey" --value "your-key"
az keyvault secret set --vault-name "nba-keyvault" --name "ApiBasketballKey" --value "your-key"

# Grant Function App access
az functionapp identity assign --name "green-bier-sports-nba" --resource-group "nba-resources"
$principalId = az functionapp identity show --name "green-bier-sports-nba" --resource-group "nba-resources" --query principalId -o tsv
az keyvault set-policy --name "nba-keyvault" --object-id $principalId --secret-permissions get

# Reference Key Vault in app settings
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings `
        THE_ODDS_API_KEY="@Microsoft.KeyVault(SecretUri=https://nba-keyvault.vault.azure.net/secrets/TheOddsApiKey/)" `
        API_BASKETBALL_KEY="@Microsoft.KeyVault(SecretUri=https://nba-keyvault.vault.azure.net/secrets/ApiBasketballKey/)"
```

### 3. Deploy Function Code

Deploy the function code to Azure:

```powershell
.\azure\deploy.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources"
```

### 4. Upload Model Files

The prediction models need to be available in Azure. Options:

**Option A: Include in deployment (for small models)**
- Models are copied during deployment

**Option B: Azure Blob Storage (Recommended)**
```powershell
# Create storage container
az storage container create --name "nba-models" --account-name "your-storage-account"

# Upload models
az storage blob upload-batch `
    --destination "nba-models" `
    --source "data/processed/models" `
    --account-name "your-storage-account"

# Update app setting to use blob storage
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings MODELS_STORAGE_CONNECTION="DefaultEndpointsProtocol=https;AccountName=..."
```

### 5. Test the Functions

Test each endpoint:

```powershell
# Generate picks for today
Invoke-RestMethod -Uri "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today" -Method Get

# Generate picks for specific game
Invoke-RestMethod -Uri "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today&matchup=Lakers" -Method Get

# View live tracker
Start-Process "https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today"
```

## Microsoft Teams Integration

### 1. Register Teams Bot

1. Go to [Azure Portal](https://portal.azure.com)
2. Create an "Azure Bot" resource
3. Configure:
   - Bot handle: `nba-picks-bot`
   - Subscription: Your subscription
   - Resource group: `nba-resources`
   - Pricing tier: F0 (Free) or S1
   - Microsoft App ID: Will be generated
4. Note the **Microsoft App ID** and **Password**

### 2. Configure Bot Endpoint

In Azure Bot settings, set:
- **Messaging endpoint**: `https://green-bier-sports-nba.azurewebsites.net/api/teams/bot`

### 3. Configure Teams Channel

1. In Azure Bot → Channels
2. Click "Microsoft Teams"
3. Click "Apply"
4. Copy the **Teams App ID**

### 4. Create Teams App Manifest

Create a Teams app package:

1. Go to [App Studio](https://teams.microsoft.com/l/app/MicrosoftTeamsAppStudio) in Teams
2. Create new app
3. Configure:
   - App details (name, description, icons)
   - Bot configuration:
     - Bot ID: Your Microsoft App ID
     - Scope: Personal, Team, Group Chat
   - Commands:
     - Command: `picks`
     - Description: `Generate NBA picks`
     - Parameters: `date` (optional), `matchup` (optional)
4. Download app package
5. Upload to Teams (App Studio → Publish)

### 5. Install Bot in Teams

1. In Teams, go to Apps
2. Find your bot
3. Click "Add" to install in personal, team, or channel

### 6. Use the Bot

Once installed, you can use commands:

```
@nba-picks-bot run picks for today
@nba-picks-bot run picks for tomorrow
@nba-picks-bot run picks for Lakers
@nba-picks-bot run picks for Lakers vs Celtics
```

## Live Tracker in Teams

### Option 1: Web Tab in Teams Channel

1. In your Teams channel, click "+" to add a tab
2. Select "Website"
3. URL: `https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today`
4. Name: "NBA Picks Tracker"
5. Click "Save"

### Option 2: Adaptive Card with Link

The Teams bot can send an Adaptive Card with a link to the tracker. Users click the link to view the live dashboard.

## Monitoring

### Application Insights

The Function App is automatically configured with Application Insights:

1. Go to Azure Portal → Function App → Application Insights
2. View:
   - Request rates and failures
   - Performance metrics
   - Logs and traces
   - Live metrics

### Function Logs

View real-time logs:

```powershell
az functionapp log tail --name "green-bier-sports-nba" --resource-group "nba-resources"
```

## Cost Optimization

1. **Consumption Plan**: Default is Pay-per-execution (very cost-effective)
2. **Cold Start**: First request after inactivity may take longer (5-30s). Consider:
   - Premium Plan (always-warm instances) if low latency is critical
   - Note: This does NOT affect manual initiation - picks are still triggered manually
3. **Model Caching**: Models are loaded once per instance (shared across functions) - improves performance but doesn't affect manual control

## Troubleshooting

### Function Timeout

Default timeout is 10 minutes. For longer operations:
- Increase timeout in `host.json`: `"functionTimeout": "00:15:00"`
- Consider breaking into multiple functions
- Use Durable Functions for long-running workflows

### Import Errors

If you see import errors:
1. Check that all dependencies are in `requirements.txt`
2. Verify Python version matches (3.11)
3. Check function logs for specific error

### Teams Bot Not Responding

1. Verify bot endpoint is accessible: `curl https://your-function-app.azurewebsites.net/api/teams/bot`
2. Check bot authentication (Microsoft App ID/Password)
3. Verify Teams channel is configured in Azure Bot
4. Check Application Insights for errors

## Design Philosophy

**Manual Initiation Only** - All picks are manually triggered:
- Call the API endpoint when you want picks
- Use Teams bot commands: `@nba-picks-bot run picks for today`
- Request picks through the Teams interface

No automated scheduling - you control when picks are generated.

## Next Steps

1. **Add Authentication**: Secure endpoints with Azure AD
2. **Database Integration**: Store picks in Azure SQL/Cosmos DB for history tracking
3. **Live Score Updates**: Integrate live score API for real-time pick tracking
4. **Notifications**: Send Teams notifications when picks are ready (manual trigger)
5. **Enhanced Tracking**: Improve live tracker with real-time score updates

## Support

For issues or questions:
- Check Application Insights logs
- Review Azure Function logs
- Check Teams Bot logs in Azure Portal