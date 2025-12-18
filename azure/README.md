# Azure Function App - Green Bier Sports NBA

This directory contains the Azure Function App deployment for the NBA prediction system with Microsoft Teams integration.

## Quick Start

### 1. Setup Azure Resources

```powershell
.\azure\setup_function_app.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources" `
    -Location "eastus"
```

### 2. Configure API Keys

```powershell
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings `
        THE_ODDS_API_KEY="your-key" `
        API_BASKETBALL_KEY="your-key"
```

### 3. Deploy Functions

```powershell
.\azure\deploy.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources"
```

## Functions

### 1. generate_picks

**Endpoint:** `GET/POST /api/generate_picks`

Generates NBA picks for a specific date or game.

**Query Parameters:**
- `date`: `today`, `tomorrow`, or `YYYY-MM-DD` (default: `today`)
- `matchup`: Optional team name filter (e.g., `Lakers`)
- `format`: `json` (default) or `teams` (Adaptive Card format)
- `channel_id`: Optional Teams channel ID to post results

**Examples:**
```bash
# Get today's picks
curl "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today"

# Get picks for specific game
curl "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today&matchup=Lakers"

# Get picks and post to Teams channel
curl "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today&channel_id=19:channel-id@thread.tacv2"
```

**Response:**
```json
{
  "date": "2025-12-18",
  "total_plays": 12,
  "games": 5,
  "predictions": [
    {
      "matchup": "Lakers @ Celtics",
      "home_team": "Celtics",
      "away_team": "Lakers",
      "commence_time": "2025-12-18T19:00:00Z",
      "plays": [
        {
          "period": "FG",
          "market": "SPREAD",
          "pick": "Lakers +5.5",
          "edge": 2.3,
          "confidence": 0.65,
          "line": 5.5,
          "odds": -110
        }
      ],
      "play_count": 1
    }
  ],
  "generated_at": "2025-12-18T10:00:00"
}
```

### 2. teams_bot

**Endpoint:** `POST /api/teams/bot`

Microsoft Teams bot endpoint that handles bot commands.

**Supported Commands:**
- `run picks for today`
- `run picks for tomorrow`
- `run picks for Lakers`
- `run picks for Lakers vs Celtics`

Returns Adaptive Card formatted response suitable for Teams.

### 3. live_tracker

**Endpoint:** `GET /api/live_tracker`

Returns HTML dashboard for live pick tracking.

**Query Parameters:**
- `date`: `today`, `tomorrow`, or `YYYY-MM-DD` (default: `today`)
- `auto_refresh`: Refresh interval in seconds (default: `60`)

**Example:**
```
https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today&auto_refresh=60
```

## Microsoft Teams Integration

### Setting Up Teams Bot

1. **Register Bot in Azure Portal:**
   - Create "Azure Bot" resource
   - Note Microsoft App ID and Password
   - Configure messaging endpoint: `https://your-function-app.azurewebsites.net/api/teams/bot`

2. **Configure Teams Channel:**
   - In Azure Bot → Channels → Microsoft Teams
   - Enable Teams channel

3. **Create Teams App:**
   - Use Teams App Studio or Teams Toolkit
   - Configure bot with your Microsoft App ID
   - Publish to Teams

4. **Install in Teams:**
   - Go to Teams → Apps
   - Find your bot and install

### Using Teams Bot

Once installed, use commands in Teams:

```
@nba-picks-bot run picks for today
```

The bot will respond with formatted picks as an Adaptive Card.

### Posting to Teams Channels

You can configure the function to automatically post picks to a Teams channel:

1. **Get Channel ID:**
   - In Teams, right-click channel → Get link to channel
   - Extract the channel ID from the link

2. **Create Incoming Webhook (Recommended):**
   - In Teams channel, go to Connectors
   - Add "Incoming Webhook"
   - Copy webhook URL
   - Set app setting: `TEAMS_WEBHOOK_URL`

3. **Use Channel ID in API:**
   ```bash
   curl "https://your-function-app.azurewebsites.net/api/generate_picks?date=today&channel_id=19:channel-id@thread.tacv2"
   ```

### Live Tracker in Teams

Add the live tracker as a web tab in Teams:

1. In Teams channel, click "+" to add tab
2. Select "Website"
3. URL: `https://your-function-app.azurewebsites.net/api/live_tracker?date=today`
4. Name: "NBA Picks Tracker"
5. Save

The tracker will auto-refresh every 60 seconds (configurable).

## Local Development

### Prerequisites

```powershell
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4 --unsafe-perm true

# Install Python dependencies
cd azure/function_app
pip install -r requirements.txt
```

### Run Locally

```powershell
# Start local function host
func start

# Functions will be available at:
# - http://localhost:7071/api/generate_picks
# - http://localhost:7071/api/teams/bot
# - http://localhost:7071/api/live_tracker
```

### Configure Local Settings

Copy `local.settings.json.example` to `local.settings.json` and fill in your API keys:

```json
{
  "Values": {
    "THE_ODDS_API_KEY": "your-key",
    "API_BASKETBALL_KEY": "your-key"
  }
}
```

## Project Structure

```
azure/
├── function_app/
│   ├── generate_picks/          # Generate picks function
│   │   ├── __init__.py
│   │   └── function.json
│   ├── teams_bot/               # Teams bot endpoint
│   │   ├── __init__.py
│   │   └── function.json
│   ├── live_tracker/            # Live tracker HTML
│   │   ├── __init__.py
│   │   └── function.json
│   ├── teams_message_service.py # Teams messaging service
│   ├── host.json                # Function App configuration
│   ├── requirements.txt         # Python dependencies
│   └── local.settings.json.example
├── Dockerfile.function          # Container deployment (optional)
├── setup_function_app.ps1      # Azure resource setup
├── deploy.ps1                   # Deployment script
└── DEPLOYMENT.md               # Detailed deployment guide
```

## Configuration

### Required App Settings

- `THE_ODDS_API_KEY`: The Odds API key
- `API_BASKETBALL_KEY`: API Basketball key

### Optional App Settings

- `ACTION_NETWORK_USERNAME`: Action Network username
- `ACTION_NETWORK_PASSWORD`: Action Network password
- `TEAMS_WEBHOOK_URL`: Teams Incoming Webhook URL
- `TEAMS_BOT_ID`: Microsoft App ID (for bot messaging)
- `TEAMS_BOT_PASSWORD`: Microsoft App Password
- `TEAMS_TENANT_ID`: Azure AD Tenant ID

### Using Azure Key Vault (Recommended)

For production, store secrets in Azure Key Vault:

```powershell
# Create Key Vault
az keyvault create --name "nba-keyvault" --resource-group "nba-resources"

# Store secrets
az keyvault secret set --vault-name "nba-keyvault" --name "TheOddsApiKey" --value "your-key"

# Reference in app settings
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings `
        THE_ODDS_API_KEY="@Microsoft.KeyVault(SecretUri=https://nba-keyvault.vault.azure.net/secrets/TheOddsApiKey/)"
```

## Monitoring

- **Application Insights**: Automatically configured
- **Function Logs**: View in Azure Portal or via CLI
- **Metrics**: Monitor in Azure Portal → Function App → Metrics

## Cost

- **Consumption Plan**: Pay per execution (very cost-effective)
- **Estimated Cost**: ~$0.20 per million function executions
- **Cold Start**: First request may take 5-30s

For production with low latency requirements, consider:
- **Premium Plan**: Always-warm instances
- **Dedicated Plan**: Predictable costs

## Troubleshooting

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed troubleshooting guide.

## Design Philosophy

**All picks are manually initiated** - no automated scheduling. Picks are generated only when:
- You call the API endpoint directly
- You send a command to the Teams bot
- You request picks through the Teams interface

This gives you full control over when picks are generated.

## Next Steps

1. Add database integration (Azure SQL/Cosmos DB) to store picks history
2. Implement live score updates for real-time tracking
3. Add authentication/authorization for secure access
4. Set up alerts and notifications for pick status updates