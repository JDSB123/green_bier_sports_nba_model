# Green Bier Sports Ventures - Enterprise Deployment Script
# 
# This script deploys the full enterprise infrastructure:
#   1. Shared resources (Key Vault, ACR, Container Apps Environment)
#   2. NBA-specific resources
#
# Usage:
#   .\deploy-enterprise.ps1 -TheOddsApiKey "<key>" -ApiBasketballKey "<key>"
#
# For NCAAM (run from NCAAM workspace):
#   .\deploy-sport.ps1 -Sport ncaam

param(
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "prod",
    
    [Parameter(Mandatory=$true)]
    [string]$TheOddsApiKey,
    
    [Parameter(Mandatory=$false)]
    [string]$ApiBasketballKey = "",
    
    [Parameter(Mandatory=$false)]
    [string]$TeamsWebhookUrl = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$SharedOnly,
    
    [Parameter(Mandatory=$false)]
    [switch]$NbaOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║       GREEN BIER SPORTS VENTURES - Enterprise Deploy         ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Check Azure CLI
Write-Host "[1/6] Checking Azure CLI..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "  → Not logged in. Running 'az login'..." -ForegroundColor Yellow
    az login
    $account = az account show | ConvertFrom-Json
}
Write-Host "  ✓ Logged in: $($account.user.name)" -ForegroundColor Green
Write-Host "  ✓ Subscription: $($account.name)" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 2: Deploy Shared Infrastructure
# ============================================================================
if (-not $NbaOnly) {
    Write-Host "[2/6] Deploying SHARED infrastructure..." -ForegroundColor Yellow
    Write-Host "  → Resource Group: gbs-shared-rg" -ForegroundColor Cyan
    
    az group create --name "gbs-shared-rg" --location $Location --output none
    
    $sharedResult = az deployment group create `
        --resource-group "gbs-shared-rg" `
        --name "gbs-shared-$(Get-Date -Format 'yyyyMMdd-HHmmss')" `
        --template-file "$ScriptDir\shared\main.bicep" `
        --parameters environment=$Environment location=$Location `
        --output json | ConvertFrom-Json
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Shared deployment failed!" -ForegroundColor Red
        exit 1
    }
    
    $acrName = $sharedResult.properties.outputs.containerRegistryName.value
    $acrServer = $sharedResult.properties.outputs.containerRegistryLoginServer.value
    $keyVaultName = $sharedResult.properties.outputs.keyVaultName.value
    
    Write-Host "  ✓ Container Registry: $acrServer" -ForegroundColor Green
    Write-Host "  ✓ Key Vault: $keyVaultName" -ForegroundColor Green
    Write-Host ""
    
    # Add secrets to Key Vault
    Write-Host "[3/6] Adding secrets to Key Vault..." -ForegroundColor Yellow
    
    az keyvault secret set --vault-name $keyVaultName --name "THE-ODDS-API-KEY" --value $TheOddsApiKey --output none
    Write-Host "  ✓ THE-ODDS-API-KEY" -ForegroundColor Green
    
    if ($ApiBasketballKey) {
        az keyvault secret set --vault-name $keyVaultName --name "API-BASKETBALL-KEY" --value $ApiBasketballKey --output none
        Write-Host "  ✓ API-BASKETBALL-KEY" -ForegroundColor Green
    }
    
    if ($TeamsWebhookUrl) {
        az keyvault secret set --vault-name $keyVaultName --name "TEAMS-WEBHOOK-URL" --value $TeamsWebhookUrl --output none
        Write-Host "  ✓ TEAMS-WEBHOOK-URL" -ForegroundColor Green
    }
    Write-Host ""
} else {
    Write-Host "[2/6] Skipping shared (--NbaOnly)" -ForegroundColor Gray
    Write-Host "[3/6] Skipping secrets (--NbaOnly)" -ForegroundColor Gray
    $acrServer = "gbssportsacr.azurecr.io"
    $acrName = "gbssportsacr"
}

if ($SharedOnly) {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║              SHARED INFRASTRUCTURE COMPLETE                  ║" -ForegroundColor Green
    Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next: Deploy sport-specific resources:" -ForegroundColor Yellow
    Write-Host "  .\deploy-enterprise.ps1 -NbaOnly -TheOddsApiKey <key>" -ForegroundColor White
    exit 0
}

# ============================================================================
# STEP 4: Deploy NBA Infrastructure
# ============================================================================
Write-Host "[4/6] Deploying NBA infrastructure..." -ForegroundColor Yellow
Write-Host "  → Resource Group: gbs-nba-rg" -ForegroundColor Cyan

az group create --name "gbs-nba-rg" --location $Location --output none

$nbaResult = az deployment group create `
    --resource-group "gbs-nba-rg" `
    --name "gbs-nba-$(Get-Date -Format 'yyyyMMdd-HHmmss')" `
    --template-file "$ScriptDir\nba\main.bicep" `
    --parameters environment=$Environment location=$Location sharedResourceGroup="gbs-shared-rg" `
    --output json | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ NBA deployment failed!" -ForegroundColor Red
    exit 1
}

$nbaApiUrl = $nbaResult.properties.outputs.containerAppUrl.value
$nbaFunctionUrl = $nbaResult.properties.outputs.functionAppUrl.value
$nbaStorageAccount = $nbaResult.properties.outputs.storageAccountName.value

Write-Host "  ✓ Container App: $nbaApiUrl" -ForegroundColor Green
Write-Host "  ✓ Function App: $nbaFunctionUrl" -ForegroundColor Green
Write-Host "  ✓ Storage: $nbaStorageAccount" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 5: Build and Push NBA Container
# ============================================================================
Write-Host "[5/6] Building and pushing NBA container..." -ForegroundColor Yellow

az acr login --name $acrName
$imageName = "$acrServer/nba-model:v5.1"
$imageLatest = "$acrServer/nba-model:latest"

docker build -t $imageName -t $imageLatest -f "$ScriptDir\..\Dockerfile" "$ScriptDir\.."
docker push $imageName
docker push $imageLatest

Write-Host "  ✓ Pushed: $imageName" -ForegroundColor Green
Write-Host ""

# ============================================================================
# STEP 6: Update Container App
# ============================================================================
Write-Host "[6/6] Updating Container App with new image..." -ForegroundColor Yellow

az containerapp update `
    --name "gbs-nba-api" `
    --resource-group "gbs-nba-rg" `
    --image $imageName `
    --output none

Write-Host "  ✓ Container App updated" -ForegroundColor Green
Write-Host ""

# ============================================================================
# DONE
# ============================================================================
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           GREEN BIER SPORTS - DEPLOYMENT COMPLETE            ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "🏀 NBA API:      $nbaApiUrl" -ForegroundColor Cyan
Write-Host "⚡ NBA Trigger:  $nbaFunctionUrl" -ForegroundColor Cyan
Write-Host "📦 NBA Storage:  $nbaStorageAccount" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test:" -ForegroundColor Yellow
Write-Host "  curl $nbaApiUrl/health" -ForegroundColor White
Write-Host "  curl $nbaApiUrl/slate/today" -ForegroundColor White
Write-Host ""
Write-Host "Deploy NCAAM (from NCAAM workspace):" -ForegroundColor Yellow
Write-Host "  az deployment group create -g gbs-ncaam-rg -f infra/ncaam/main.bicep \" -ForegroundColor White
Write-Host "    --parameters sharedResourceGroup=gbs-shared-rg" -ForegroundColor White
Write-Host ""
