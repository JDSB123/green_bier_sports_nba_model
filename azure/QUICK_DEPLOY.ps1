# Quick Deploy - Uses your existing resource group
# Usage: .\azure\QUICK_DEPLOY.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "green-bier-sports-nba",
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "green-bier-sport-ventures-rg",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "southcentralus"
)

Write-Host "[QUICK DEPLOY] Green Bier Sports NBA" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# Read API keys from .env
$envFile = Join-Path $PSScriptRoot "..\.env"
$TheOddsApiKey = ""
$ApiBasketballKey = ""

if (Test-Path $envFile) {
    Write-Host "[STEP] Reading API keys from .env..." -ForegroundColor Yellow
    $envLines = Get-Content $envFile -ErrorAction SilentlyContinue
    foreach ($envLine in $envLines) {
        $cleanLine = $envLine.Trim()
        if ($cleanLine -ne "" -and -not $cleanLine.StartsWith("#") -and $cleanLine.Contains("=")) {
            $idx = $cleanLine.IndexOf("=")
            if ($idx -gt 0) {
                $envKey = $cleanLine.Substring(0, $idx).Trim()
                $envValue = $cleanLine.Substring($idx + 1).Trim()
                if ($envKey -eq "THE_ODDS_API_KEY") { $TheOddsApiKey = $envValue }
                if ($envKey -eq "API_BASKETBALL_KEY") { $ApiBasketballKey = $envValue }
            }
        }
    }
    if ($TheOddsApiKey -or $ApiBasketballKey) {
        Write-Host "[OK] Found API keys" -ForegroundColor Green
    }
}
Write-Host ""

# Check if function app exists
Write-Host "[STEP] Checking Function App..." -ForegroundColor Yellow
$faExists = az functionapp show --name $FunctionAppName --resource-group $ResourceGroup 2>$null
if (-not $faExists) {
    Write-Host "[CREATE] Function App does not exist. Creating..." -ForegroundColor Yellow
    Write-Host "  This will take 2-3 minutes..." -ForegroundColor Gray
    
    # Generate storage account name
    $storageName = "$($FunctionAppName.Replace('-', ''))stor".Substring(0, [Math]::Min(24, "$($FunctionAppName.Replace('-', ''))stor".Length))
    
    # Create storage account if needed
    $storageExists = az storage account show --name $storageName --resource-group $ResourceGroup 2>$null
    if (-not $storageExists) {
        az storage account create --name $storageName --resource-group $ResourceGroup --location $Location --sku Standard_LRS | Out-Null
    }
    
    # Create App Service Plan if needed
    $planName = "$FunctionAppName-plan"
    $planExists = az functionapp plan show --name $planName --resource-group $ResourceGroup 2>$null
    if (-not $planExists) {
        az functionapp plan create --name $planName --resource-group $ResourceGroup --location $Location --sku Y1 --is-linux | Out-Null
    }
    
    # Create Function App
    az functionapp create `
        --name $FunctionAppName `
        --resource-group $ResourceGroup `
        --storage-account $storageName `
        --plan $planName `
        --runtime python `
        --runtime-version 3.11 `
        --functions-version 4 `
        --os-type Linux | Out-Null
    
    Write-Host "[OK] Function App created" -ForegroundColor Green
} else {
    Write-Host "[OK] Function App already exists" -ForegroundColor Green
}
Write-Host ""

# Configure API keys
Write-Host "[STEP] Configuring API keys..." -ForegroundColor Yellow
$settings = @("TEAMS_TENANT_ID=18ee0910-417d-4a81-a3f5-7945bdbd5a78")
if ($TheOddsApiKey) { $settings += "THE_ODDS_API_KEY=$TheOddsApiKey" }
if ($ApiBasketballKey) { $settings += "API_BASKETBALL_KEY=$ApiBasketballKey" }

az functionapp config appsettings set `
    --name $FunctionAppName `
    --resource-group $ResourceGroup `
    --settings $settings 2>&1 | Out-Null

Write-Host "[OK] App settings configured" -ForegroundColor Green
Write-Host ""

# Deploy functions
Write-Host "[STEP] Deploying functions..." -ForegroundColor Yellow
Write-Host "  This will take 3-5 minutes..." -ForegroundColor Gray
$functionAppDir = Join-Path $PSScriptRoot "function_app"
Push-Location $functionAppDir
func azure functionapp publish $FunctionAppName --python 2>&1 | Tee-Object -Variable deployOutput
$deploySuccess = $LASTEXITCODE -eq 0
Pop-Location

if ($deploySuccess) {
    Write-Host "[OK] Deployment successful!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Deployment had issues. Check output above." -ForegroundColor Red
}
Write-Host ""

# Summary
$functionUrl = "https://${FunctionAppName}.azurewebsites.net"
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "[SUCCESS] Deployment Complete!" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "Function App URL: $functionUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "Endpoints:" -ForegroundColor Yellow
Write-Host "  Generate Picks: ${functionUrl}/api/generate_picks" -ForegroundColor White
Write-Host "  Live Tracker: ${functionUrl}/api/live_tracker" -ForegroundColor White
Write-Host ""
Write-Host "Test it:" -ForegroundColor Yellow
Write-Host "  .\azure\test_picks.ps1 -FunctionAppName $FunctionAppName" -ForegroundColor Gray
Write-Host ""