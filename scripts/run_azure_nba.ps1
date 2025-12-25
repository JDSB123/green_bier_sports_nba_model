# Quick Run Azure NBA Deployment
# Simple script to deploy/update your Azure NBA setup
#
# Usage:
#   .\scripts\run_azure_nba.ps1                    # Deploy/update current setup
#   .\scripts\run_azure_nba.ps1 -Tag v6.10         # Deploy specific version
#   .\scripts\run_azure_nba.ps1 -NewEnvironment    # Deploy to new resource group

Param(
    [string]$Tag = "v6.10",
    [switch]$NewEnvironment,
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$Location = "eastus"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Azure NBA - Quick Deploy" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Check Azure CLI
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Logging in to Azure..." -ForegroundColor Yellow
    az login
    $account = az account show | ConvertFrom-Json
}
Write-Host "Azure Subscription: $($account.name)" -ForegroundColor Cyan
Write-Host ""

# Check if resource group exists
$rgExists = az group show --name $ResourceGroup 2>$null
if (-not $rgExists -or $NewEnvironment) {
    Write-Host "Resource group '$ResourceGroup' not found or creating new environment..." -ForegroundColor Yellow
    Write-Host "Running full provisioning..." -ForegroundColor Cyan
    Write-Host ""
    
    $provisionScript = Join-Path $PSScriptRoot "..\infra\nba\create-nba-gbsv-model-rg.ps1"
    if (Test-Path $provisionScript) {
        & $provisionScript -ResourceGroup $ResourceGroup -Location $Location -Tag $Tag
    } else {
        Write-Host "ERROR: Provisioning script not found at: $provisionScript" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Resource group exists. Updating deployment..." -ForegroundColor Cyan
    Write-Host ""
    
    # Use the deploy script
    $deployScript = Join-Path $PSScriptRoot "..\infra\nba\deploy.ps1"
    if (Test-Path $deployScript) {
        & $deployScript -ResourceGroup $ResourceGroup -Tag $Tag
    } else {
        Write-Host "ERROR: Deploy script not found at: $deployScript" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Deployment Complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Get API URL
$fqdn = az containerapp show -n nba-gbsv-api -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv 2>$null
if ($fqdn) {
    Write-Host "API URL: https://$fqdn" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Testing health endpoint..." -ForegroundColor Yellow
    try {
        $health = Invoke-WebRequest -Uri "https://$fqdn/health" -UseBasicParsing -TimeoutSec 10
        Write-Host "  ✓ API is healthy!" -ForegroundColor Green
        Write-Host "  Response: $($health.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Health check failed (API may still be starting)" -ForegroundColor Yellow
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Gray
    }
} else {
    Write-Host "Could not retrieve API URL. Container App may not be ready yet." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  # View logs" -ForegroundColor White
Write-Host "  az containerapp logs show -n nba-gbsv-api -g $ResourceGroup --follow" -ForegroundColor Gray
Write-Host ""
Write-Host "  # Get API URL" -ForegroundColor White
Write-Host "  az containerapp show -n nba-gbsv-api -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv" -ForegroundColor Gray
Write-Host ""

