# Quick Start - Viewing & Using the Azure Function App

## What Was Created

I've created a complete Azure Function App setup in the `azure/` directory. Here's what you have:

### Files Created:
- `azure/function_app/` - All the function code
  - `generate_picks/` - Function to generate NBA picks
  - `teams_bot/` - Microsoft Teams bot integration
  - `live_tracker/` - HTML dashboard for live tracking
- `azure/setup_function_app.ps1` - Script to create Azure resources
- `azure/deploy.ps1` - Script to deploy functions to Azure
- `azure/README.md` - Full documentation
- `azure/DEPLOYMENT.md` - Detailed deployment guide

## Viewing the Code

### 1. Browse the Files

Open these files to see what was created:

```
azure/
├── function_app/
│   ├── generate_picks/
│   │   ├── __init__.py          ← Main picks generation logic
│   │   └── function.json        ← Function configuration
│   ├── teams_bot/
│   │   ├── __init__.py          ← Teams bot handler
│   │   └── function.json
│   ├── live_tracker/
│   │   ├── __init__.py          ← HTML dashboard generator
│   │   └── function.json
│   ├── teams_message_service.py ← Service for posting to Teams
│   ├── host.json                ← Function App config
│   └── requirements.txt         ← Python dependencies
├── README.md                    ← Quick reference
├── DEPLOYMENT.md                ← Detailed guide
└── QUICK_START.md               ← This file
```

### 2. Key Files to Review

**Main Picks Function:**
- `azure/function_app/generate_picks/__init__.py` - This is the core function that generates picks

**Teams Integration:**
- `azure/function_app/teams_bot/__init__.py` - Handles Teams bot commands
- `azure/function_app/teams_message_service.py` - Posts results to Teams

**Live Tracker:**
- `azure/function_app/live_tracker/__init__.py` - Generates HTML dashboard

## Testing Locally (Before Deploying)

### Step 1: Install Prerequisites

```powershell
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4 --unsafe-perm true

# Install Python dependencies
cd azure/function_app
pip install -r requirements.txt
```

### Step 2: Configure Local Settings

```powershell
# Copy the example settings file
cd azure/function_app
Copy-Item local.settings.json.example local.settings.json

# Edit local.settings.json and add your API keys:
# {
#   "Values": {
#     "THE_ODDS_API_KEY": "your-key-here",
#     "API_BASKETBALL_KEY": "your-key-here"
#   }
# }
```

### Step 3: Run Locally

```powershell
# From azure/function_app directory
func start
```

You'll see output like:
```
Functions:
        generate_picks: [GET,POST] http://localhost:7071/api/generate_picks
        teams_bot: [GET,POST] http://localhost:7071/api/teams/bot
        live_tracker: [GET] http://localhost:7071/api/live_tracker
```

### Step 4: Test the Functions

**Test Generate Picks:**
```powershell
# Open in browser or use PowerShell
Invoke-RestMethod -Uri "http://localhost:7071/api/generate_picks?date=today" | ConvertTo-Json
```

**Test Live Tracker:**
```powershell
# Opens in browser
Start-Process "http://localhost:7071/api/live_tracker?date=today"
```

## Deploying to Azure

### Step 1: Login to Azure

```powershell
az login
```

### Step 2: Create Azure Resources

```powershell
# From project root
.\azure\setup_function_app.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources" `
    -Location "eastus"
```

This creates:
- Resource Group
- Storage Account
- App Service Plan
- Function App
- Application Insights

### Step 3: Set API Keys

```powershell
az functionapp config appsettings set `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources" `
    --settings `
        THE_ODDS_API_KEY="your-odds-api-key" `
        API_BASKETBALL_KEY="your-basketball-api-key"
```

### Step 4: Deploy Functions

```powershell
.\azure\deploy.ps1 `
    -FunctionAppName "green-bier-sports-nba" `
    -ResourceGroup "nba-resources"
```

### Step 5: Test Deployed Functions

**Generate Picks:**
```powershell
Invoke-RestMethod -Uri "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today"
```

**View Live Tracker:**
```
https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today
```

## Using the Functions

### Option 1: Direct API Calls

```powershell
# Get today's picks
Invoke-RestMethod -Uri "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today"

# Get picks for specific team
Invoke-RestMethod -Uri "https://green-bier-sports-nba.azurewebsites.net/api/generate_picks?date=today&matchup=Lakers"

# View live tracker in browser
Start-Process "https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today"
```

### Option 2: Microsoft Teams Bot

Once you've configured the Teams bot (see DEPLOYMENT.md for details), you can use:

```
@nba-picks-bot run picks for today
@nba-picks-bot run picks for Lakers
```

### Option 3: Teams Web Tab

Add the live tracker as a tab in Teams:

1. Go to your Teams channel
2. Click "+" to add tab
3. Select "Website"
4. URL: `https://green-bier-sports-nba.azurewebsites.net/api/live_tracker?date=today`
5. Name it "NBA Picks Tracker"

## Viewing Logs

### Local Logs
When running `func start`, logs appear in the terminal.

### Azure Logs
```powershell
# Stream live logs
az functionapp log tail `
    --name "green-bier-sports-nba" `
    --resource-group "nba-resources"

# View in Azure Portal
# Go to: Function App → Functions → Monitor
```

## Next Steps

1. **Test locally first** - Make sure everything works before deploying
2. **Deploy to Azure** - Use the deployment scripts
3. **Set up Teams Bot** - Follow instructions in DEPLOYMENT.md
4. **Configure Models** - Upload your prediction models (see DEPLOYMENT.md section 4)

## Need Help?

- Check `azure/README.md` for detailed function documentation
- Check `azure/DEPLOYMENT.md` for deployment troubleshooting
- View function logs for errors
- Check Application Insights in Azure Portal for monitoring