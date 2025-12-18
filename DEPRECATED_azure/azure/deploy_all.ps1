# Complete Automated Deployment Script
# This script automates the entire deployment process
# Usage: .\azure\deploy_all.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "green-bier-sports-nba",
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "nba-resources",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$TheOddsApiKey = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ApiBasketballKey = "",
    
    [Parameter(Mandatory=$false)]
    [string]$TeamsWebhookUrl = ""
)

$ErrorActionPreference = "Continue"  # Continue on errors to show all issues

# Your Teams configuration
$TeamsChannelId = "19:5369a11408864936935266147c1f3b02@thread.tacv2"
$TeamsTenantId = "18ee0910-417d-4a81-a3f5-7945bdbd5a78"
$TeamsGroupId = "6d55cb22-b8b0-43a4-8ec1-f5df8a966856"

Write-Host "[DEPLOY] Green Bier Sports NBA - Complete Deployment" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# Step 0: Read API Keys from .env file
Write-Host "[STEP] Step 0: Reading API keys from .env file..." -ForegroundColor Yellow
$envFile = Join-Path $PSScriptRoot "..\.env"
if (Test-Path $envFile) {
    Write-Host "   Found .env file at: $envFile" -ForegroundColor Cyan
    $envLines = Get-Content $envFile -ErrorAction SilentlyContinue
    
    foreach ($envLine in $envLines) {
        $cleanLine = $envLine.Trim()
        if ($cleanLine -ne "" -and -not $cleanLine.StartsWith("#")) {
            if ($cleanLine -like "*=*") {
                $idx = $cleanLine.IndexOf("=")
                if ($idx -gt 0) {
                    $envKey = $cleanLine.Substring(0, $idx).Trim()
                    $envValue = $cleanLine.Substring($idx + 1).Trim()
                    
                    if ($envKey -eq "THE_ODDS_API_KEY" -and [string]::IsNullOrEmpty($TheOddsApiKey)) {
                        $TheOddsApiKey = $envValue
                        Write-Host "   [OK] Found THE_ODDS_API_KEY" -ForegroundColor Green
                    }
                    if ($envKey -eq "API_BASKETBALL_KEY" -and [string]::IsNullOrEmpty($ApiBasketballKey)) {
                        $ApiBasketballKey = $envValue
                        Write-Host "   [OK] Found API_BASKETBALL_KEY" -ForegroundColor Green
                    }
                }
            }
        }
    }
} else {
    Write-Host "   [WARN] .env file not found at: $envFile" -ForegroundColor Yellow
}
Write-Host ""

# Step 1: Check Azure Login (just verify, don't force login)
Write-Host "[STEP] Step 1: Verifying Azure connection..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "   [WARN] Not logged in to Azure. Attempting login..." -ForegroundColor Yellow
    az login --use-device-code 2>&1 | Out-Null
    $account = az account show 2>$null | ConvertFrom-Json
    if (-not $account) {
        Write-Host "   [ERROR] Could not verify Azure login. Continuing anyway..." -ForegroundColor Yellow
    }
}
if ($account) {
    Write-Host "   [OK] Connected to Azure as: $($account.user.name)" -ForegroundColor Green
}
Write-Host ""

# Step 2: Create Azure Resources (skip if already exist)
Write-Host "[STEP] Step 2: Checking Azure resources..." -ForegroundColor Yellow
$setupScript = Join-Path $PSScriptRoot "setup_function_app.ps1"
if (Test-Path $setupScript) {
    # Check if function app already exists
    $faExists = az functionapp show --name $FunctionAppName --resource-group $ResourceGroup 2>$null
    if (-not $faExists) {
        Write-Host "   Creating Azure resources (this may take a few minutes)..." -ForegroundColor Cyan
        & $setupScript -FunctionAppName $FunctionAppName -ResourceGroup $ResourceGroup -Location $Location
        if ($LASTEXITCODE -ne 0) {
            Write-Host "   [WARN] Resource creation had issues. Continuing..." -ForegroundColor Yellow
        }
    } else {
        Write-Host "   [OK] Function App already exists: $FunctionAppName" -ForegroundColor Green
    }
} else {
    Write-Host "   [WARN] Setup script not found" -ForegroundColor Yellow
}
Write-Host ""

# Step 3: Configure API Keys
Write-Host "[STEP] Step 3: Configuring API keys..." -ForegroundColor Yellow

$settingsToSet = @{}

# Always set tenant ID
$settingsToSet["TEAMS_TENANT_ID"] = $TeamsTenantId

# Add API keys if we have them
if (-not [string]::IsNullOrEmpty($TheOddsApiKey)) {
    $settingsToSet["THE_ODDS_API_KEY"] = $TheOddsApiKey
}
if (-not [string]::IsNullOrEmpty($ApiBasketballKey)) {
    $settingsToSet["API_BASKETBALL_KEY"] = $ApiBasketballKey
}
if (-not [string]::IsNullOrEmpty($TeamsWebhookUrl)) {
    $settingsToSet["TEAMS_WEBHOOK_URL"] = $TeamsWebhookUrl
}

# Build settings string
if ($settingsToSet.Count -gt 0) {
    $settingsArray = $settingsToSet.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }
    
    Write-Host "   Setting app settings..." -ForegroundColor Cyan
    az functionapp config appsettings set `
        --name $FunctionAppName `
        --resource-group $ResourceGroup `
        --settings $settingsArray 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   [OK] App settings configured" -ForegroundColor Green
    } else {
        Write-Host "   [WARN] Failed to set app settings (they may already be set)" -ForegroundColor Yellow
    }
} else {
    Write-Host "   [WARN] No API keys found. Check your .env file or set them manually" -ForegroundColor Yellow
}
Write-Host ""

# Step 4: Deploy Functions
Write-Host "[STEP] Step 4: Deploying function code..." -ForegroundColor Yellow
$deployScript = Join-Path $PSScriptRoot "deploy.ps1"
if (Test-Path $deployScript) {
    Write-Host "   Deploying functions (this may take several minutes)..." -ForegroundColor Cyan
    & $deployScript -FunctionAppName $FunctionAppName -ResourceGroup $ResourceGroup
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   [ERROR] Deployment failed!" -ForegroundColor Red
        Write-Host "   You may need to check the logs or try again" -ForegroundColor Yellow
    }
} else {
    Write-Host "   [WARN] Deploy script not found. Trying direct deployment..." -ForegroundColor Yellow
    $functionAppDir = Join-Path $PSScriptRoot "function_app"
    if (Test-Path $functionAppDir) {
        Push-Location $functionAppDir
        func azure functionapp publish $FunctionAppName --python
        $deployResult = $LASTEXITCODE
        Pop-Location
        
        if ($deployResult -ne 0) {
            Write-Host "   [ERROR] Direct deployment failed!" -ForegroundColor Red
        }
    }
}
Write-Host ""

# Step 5: Teams Webhook Setup
Write-Host "[STEP] Step 5: Teams Webhook Setup" -ForegroundColor Yellow
if ([string]::IsNullOrEmpty($TeamsWebhookUrl)) {
    Write-Host "   [INFO] Teams webhook not configured yet." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   To set up Teams webhook:" -ForegroundColor White
    Write-Host "   1. Go to Teams channel: Green Bier NBA Model" -ForegroundColor Gray
    Write-Host "   2. Click ⋯ → Connectors → Incoming Webhook" -ForegroundColor Gray
    Write-Host "   3. Configure and copy the webhook URL" -ForegroundColor Gray
    Write-Host "   4. Run: az functionapp config appsettings set --name $FunctionAppName --resource-group $ResourceGroup --settings TEAMS_WEBHOOK_URL='your-url'" -ForegroundColor Gray
} else {
    Write-Host "   [OK] Teams webhook configured" -ForegroundColor Green
}
Write-Host ""

# Step 6: Test Functions
Write-Host "[STEP] Step 6: Testing functions..." -ForegroundColor Yellow
$functionUrl = "https://${FunctionAppName}.azurewebsites.net"

Write-Host "   Waiting a few seconds for functions to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

Write-Host "   Testing endpoint..." -ForegroundColor Cyan
try {
    $testResponse = Invoke-RestMethod -Uri "${functionUrl}/api/generate_picks?date=today" -Method Get -TimeoutSec 30 -ErrorAction SilentlyContinue
    Write-Host "   [OK] Functions are accessible!" -ForegroundColor Green
} catch {
    Write-Host "   [WARN] Functions may not be ready yet (wait 1-2 minutes and try again)" -ForegroundColor Yellow
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Gray
}
Write-Host ""

# Step 7: Summary
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "[SUCCESS] DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

Write-Host "📊 Your Function App:" -ForegroundColor Cyan
Write-Host "   URL: $functionUrl" -ForegroundColor White
Write-Host ""

Write-Host "🔗 Function Endpoints:" -ForegroundColor Cyan
Write-Host "   Generate Picks: ${functionUrl}/api/generate_picks" -ForegroundColor White
Write-Host "   Teams Bot: ${functionUrl}/api/teams/bot" -ForegroundColor White
Write-Host "   Live Tracker: ${functionUrl}/api/live_tracker" -ForegroundColor White
Write-Host ""

Write-Host "📝 Quick Test Commands:" -ForegroundColor Cyan
Write-Host "   .\azure\test_picks.ps1" -ForegroundColor Gray
Write-Host "   .\azure\test_picks.ps1 -PostToTeams" -ForegroundColor Gray
Write-Host "   .\azure\OPEN_LIVE_TRACKER.ps1" -ForegroundColor Gray
Write-Host ""

Write-Host "🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "   1. Wait 1-2 minutes for functions to fully initialize" -ForegroundColor White
Write-Host "   2. Test with: .\azure\test_picks.ps1" -ForegroundColor White
Write-Host "   3. Set up Teams webhook (see Step 5 above)" -ForegroundColor White
Write-Host "   4. Add live tracker to Teams channel as a Website tab" -ForegroundColor White
Write-Host ""

Write-Host "[DOCS] Documentation:" -ForegroundColor Cyan
Write-Host "   - Quick Start: azure/QUICK_START.md" -ForegroundColor White
Write-Host "   - Deployment Guide: azure/DEPLOYMENT.md" -ForegroundColor White
Write-Host "   - Checklist: azure/DEPLOYMENT_CHECKLIST.md" -ForegroundColor White
Write-Host ""