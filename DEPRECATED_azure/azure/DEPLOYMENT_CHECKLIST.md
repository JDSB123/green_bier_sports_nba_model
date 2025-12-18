# Deployment Checklist - Green Bier Sports NBA

## Your Configuration

**Teams Channel:**
- Channel Name: Green Bier NBA Model
- Channel ID: `19:5369a11408864936935266147c1f3b02@thread.tacv2`
- Group ID: `6d55cb22-b8b0-43a4-8ec1-f5df8a966856`
- Tenant ID: `18ee0910-417d-4a81-a3f5-7945bdbd5a78`
- URL: https://teams.microsoft.com/l/channel/19%3A5369a11408864936935266147c1f3b02%40thread.tacv2/Green%20Bier%20NBA%20Model?groupId=6d55cb22-b8b0-43a4-8ec1-f5df8a966856&tenantId=18ee0910-417d-4a81-a3f5-7945bdbd5a78

**Azure Status:** ✅ Linked and ready

## Quick Deployment Steps

### 1. Create Function App (If Not Already Created)

```powershell
.\azure\setup_function_app.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources" `
    -Location "eastus"
```

### 2. Set API Keys

```powershell
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings `
        THE_ODDS_API_KEY="your-odds-api-key" `
        API_BASKETBALL_KEY="your-basketball-api-key" `
        TEAMS_TENANT_ID="18ee0910-417d-4a81-a3f5-7945bdbd5a78"
```

### 3. Deploy Functions

```powershell
.\azure\deploy.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources"
```

### 4. Set Up Teams Webhook (Easiest for Posting)

1. In your Teams channel "Green Bier NBA Model"
2. Click **⋯** (three dots) → **Connectors**
3. Search for "Incoming Webhook"
4. Click **Configure**
5. Name it: "NBA Picks Bot"
6. Click **Create**
7. **Copy the webhook URL**

Then set it as an app setting:

```powershell
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings TEAMS_WEBHOOK_URL="your-webhook-url-here"
```

### 5. Test the Functions

**Test Generate Picks:**
```powershell
$functionAppName = "green-bier-sports-nba"
Invoke-RestMethod -Uri "https://${functionAppName}.azurewebsites.net/api/generate_picks?date=today" | ConvertTo-Json
```

**Post to Teams Channel:**
```powershell
$functionAppName = "green-bier-sports-nba"
$channelId = "19:5369a11408864936935266147c1f3b02@thread.tacv2"
Invoke-RestMethod -Uri "https://${functionAppName}.azurewebsites.net/api/generate_picks?date=today&channel_id=${channelId}"
```

**View Live Tracker:**
```powershell
$functionAppName = "green-bier-sports-nba"
Start-Process "https://${functionAppName}.azurewebsites.net/api/live_tracker?date=today"
```

### 6. Add Live Tracker to Teams Channel

1. Go to your Teams channel "Green Bier NBA Model"
2. Click **+** (Add a tab)
3. Select **Website**
4. URL: `https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today`
5. Name: "NBA Picks Tracker"
6. Click **Save**

## Usage Examples

### Generate Picks for Today (and Post to Teams)

```powershell
$functionAppName = "green-bier-sports-nba"
$channelId = "19:5369a11408864936935266147c1f3b02@thread.tacv2"

# This will generate picks AND post to your Teams channel
Invoke-RestMethod -Uri "https://${functionAppName}.azurewebsites.net/api/generate_picks?date=today&channel_id=${channelId}"
```

### Generate Picks for Specific Game

```powershell
$functionAppName = "green-bier-sports-nba"
$channelId = "19:5369a11408864936935266147c1f3b02@thread.tacv2"

Invoke-RestMethod -Uri "https://${functionAppName}.azurewebsites.net/api/generate_picks?date=today&matchup=Lakers&channel_id=${channelId}"
```

### View Picks JSON (Without Posting)

```powershell
$functionAppName = "green-bier-sports-nba"
Invoke-RestMethod -Uri "https://${functionAppName}.azurewebsites.net/api/generate_picks?date=today" | ConvertTo-Json -Depth 10
```

## Next Steps

1. ✅ Azure accounts linked
2. ✅ Teams channel set up
3. ⬜ Deploy Function App
4. ⬜ Configure API keys
5. ⬜ Set up Teams webhook
6. ⬜ Test functions
7. ⬜ Add live tracker to Teams

## Troubleshooting

If you get errors, check:

1. **Function App exists?**
   ```powershell
   az functionapp list --query "[].{Name:name, ResourceGroup:resourceGroup}"
   ```

2. **API keys set?**
   ```powershell
   az functionapp config appsettings list --name "green-bier-sports-nba" --resource-group "nba-resources" --query "[?name=='THE_ODDS_API_KEY' || name=='API_BASKETBALL_KEY']"
   ```

3. **Functions deployed?**
   ```powershell
   az functionapp function list --name "green-bier-sports-nba" --resource-group "nba-resources"
   ```

4. **View logs:**
   ```powershell
   az functionapp log tail --name "green-bier-sports-nba" --resource-group "nba-resources"
   ```