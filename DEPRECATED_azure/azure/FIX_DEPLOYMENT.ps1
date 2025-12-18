# Fix Deployment - Use Existing Resource Group or Create New One
# This script fixes the deployment by using your existing Azure resources

param(
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "green-bier-sports-nba",
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus"
)

$ErrorActionPreference = "Continue"

Write-Host "[FIX] Fixing Azure Function App Deployment" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# Check existing function apps
Write-Host "[CHECK] Checking existing Azure resources..." -ForegroundColor Yellow
$existingApps = az functionapp list --query "[].{Name:name, ResourceGroup:resourceGroup, State:state}" -o json | ConvertFrom-Json
$existingGroups = az group list --query "[].{Name:name, Location:location}" -o json | ConvertFrom-Json

Write-Host ""
Write-Host "Existing Function Apps:" -ForegroundColor Cyan
foreach ($app in $existingApps) {
    Write-Host "  - $($app.Name) (Resource Group: $($app.ResourceGroup))" -ForegroundColor White
}

Write-Host ""
Write-Host "Existing Resource Groups:" -ForegroundColor Cyan
foreach ($rg in $existingGroups) {
    Write-Host "  - $($rg.Name) (Location: $($rg.Location))" -ForegroundColor White
}

# Determine resource group to use
if ([string]::IsNullOrEmpty($ResourceGroup)) {
    # Check if green-bier-sport-ventures-rg exists (the one with existing function app)
    $existingRG = $existingGroups | Where-Object { $_.Name -eq "green-bier-sport-ventures-rg" }
    if ($existingRG) {
        $ResourceGroup = "green-bier-sport-ventures-rg"
        Write-Host ""
        Write-Host "[INFO] Using existing resource group: $ResourceGroup" -ForegroundColor Green
    } else {
        # Use nba-resources and create it
        $ResourceGroup = "nba-resources"
        Write-Host ""
        Write-Host "[INFO] Will create new resource group: $ResourceGroup" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "[DECISION] Target Configuration:" -ForegroundColor Cyan
Write-Host "  Function App Name: $FunctionAppName" -ForegroundColor White
Write-Host "  Resource Group: $ResourceGroup" -ForegroundColor White
Write-Host "  Location: $Location" -ForegroundColor White
Write-Host ""

# Check if resource group exists, create if not
$rgExists = az group exists --name $ResourceGroup 2>$null
if ($rgExists -eq "false") {
    Write-Host "[CREATE] Creating resource group: $ResourceGroup" -ForegroundColor Yellow
    az group create --name $ResourceGroup --location $Location | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Resource group created" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to create resource group" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[OK] Resource group exists: $ResourceGroup" -ForegroundColor Green
}

# Check if function app exists
$faExists = az functionapp show --name $FunctionAppName --resource-group $ResourceGroup 2>$null
if (-not $faExists) {
    Write-Host ""
    Write-Host "[CREATE] Function App '$FunctionAppName' does not exist" -ForegroundColor Yellow
    Write-Host "[INFO] Run the setup script to create it:" -ForegroundColor Cyan
    Write-Host "  .\azure\setup_function_app.ps1 -FunctionAppName $FunctionAppName -ResourceGroup $ResourceGroup -Location $Location" -ForegroundColor Gray
    Write-Host ""
    $create = Read-Host "Do you want to create it now? (y/n)"
    if ($create -eq "y" -or $create -eq "Y") {
        & .\azure\setup_function_app.ps1 -FunctionAppName $FunctionAppName -ResourceGroup $ResourceGroup -Location $Location
    }
} else {
    Write-Host "[OK] Function App exists: $FunctionAppName" -ForegroundColor Green
    $faInfo = az functionapp show --name $FunctionAppName --resource-group $ResourceGroup --query "{url:defaultHostName, state:state}" -o json | ConvertFrom-Json
    Write-Host "  URL: https://$($faInfo.url)" -ForegroundColor White
    Write-Host "  State: $($faInfo.state)" -ForegroundColor White
}

Write-Host ""
Write-Host "[NEXT] Next Steps:" -ForegroundColor Cyan
Write-Host "  1. If Function App was created, deploy code:" -ForegroundColor White
Write-Host "     .\azure\deploy.ps1 -FunctionAppName $FunctionAppName -ResourceGroup $ResourceGroup" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Or run full deployment:" -ForegroundColor White
Write-Host "     .\azure\deploy_all.ps1 -ResourceGroup $ResourceGroup" -ForegroundColor Gray
Write-Host ""