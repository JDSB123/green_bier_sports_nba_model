# Copy/Backup Azure NBA Configuration
# This script exports your current Azure NBA setup and optionally deploys it
#
# Usage:
#   .\scripts\copy_azure_nba.ps1                    # Export current config
#   .\scripts\copy_azure_nba.ps1 -Deploy           # Deploy to new environment
#   .\scripts\copy_azure_nba.ps1 -Backup            # Create backup of current setup

Param(
    [switch]$Deploy,
    [switch]$Backup,
    [string]$SourceResourceGroup = "nba-gbsv-model-rg",
    [string]$TargetResourceGroup = "nba-gbsv-model-rg-copy",
    [string]$Location = "eastus",
    [string]$Subscription = ""
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Azure NBA Configuration Copy/Deploy Tool" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Check Azure CLI
Write-Host "[1/5] Checking Azure CLI..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "  -> Not logged in. Running 'az login'..." -ForegroundColor Yellow
    az login
    $account = az account show | ConvertFrom-Json
}
Write-Host "  Logged in: $($account.user.name)" -ForegroundColor Green
Write-Host "  Subscription: $($account.name)" -ForegroundColor Green

if ($Subscription -and $Subscription.Trim() -ne "") {
    az account set --subscription $Subscription
    Write-Host "  Switched to subscription: $Subscription" -ForegroundColor Green
}
Write-Host ""

# Export current configuration
Write-Host "[2/5] Exporting current Azure NBA configuration..." -ForegroundColor Yellow
$exportDir = Join-Path $PSScriptRoot "..\azure_export"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$exportPath = Join-Path $exportDir "nba_config_$timestamp"

if (-not (Test-Path $exportDir)) {
    New-Item -ItemType Directory -Path $exportDir -Force | Out-Null
}

# Create the timestamped export directory
if (-not (Test-Path $exportPath)) {
    New-Item -ItemType Directory -Path $exportPath -Force | Out-Null
}

Write-Host "  Export directory: $exportPath" -ForegroundColor Cyan

# Check if source resource group exists
$rgExists = az group show --name $SourceResourceGroup 2>$null
if (-not $rgExists) {
    Write-Host "  WARNING: Source resource group '$SourceResourceGroup' not found!" -ForegroundColor Red
    Write-Host "  Creating new deployment instead..." -ForegroundColor Yellow
    $Backup = $false
    $Deploy = $true
} else {
    # Export Container App configuration
    Write-Host "  Exporting Container App configuration..." -ForegroundColor Cyan
    $containerApp = az containerapp show --name "nba-gbsv-api" --resource-group $SourceResourceGroup --output json 2>$null | ConvertFrom-Json
    if ($containerApp) {
        $containerApp | ConvertTo-Json -Depth 10 | Out-File (Join-Path $exportPath "container_app.json") -Encoding UTF8
        Write-Host "    ✓ Container App config exported" -ForegroundColor Green
    }

    # Export ACR configuration
    Write-Host "  Exporting ACR configuration..." -ForegroundColor Cyan
    $acr = az acr show --name "nbagbsacr" --resource-group $SourceResourceGroup --output json 2>$null | ConvertFrom-Json
    if ($acr) {
        $acr | ConvertTo-Json -Depth 10 | Out-File (Join-Path $exportPath "acr.json") -Encoding UTF8
        Write-Host "    ✓ ACR config exported" -ForegroundColor Green
    }

    # Export Key Vault secrets (names only, not values)
    Write-Host "  Exporting Key Vault secret names..." -ForegroundColor Cyan
    $secrets = az keyvault secret list --vault-name "nbagbs-keyvault" --query "[].name" -o json 2>$null | ConvertFrom-Json
    if ($secrets) {
        $secrets | ConvertTo-Json | Out-File (Join-Path $exportPath "keyvault_secrets.json") -Encoding UTF8
        Write-Host "    ✓ Key Vault secret names exported (values not exported for security)" -ForegroundColor Green
    }

    # Export Container Apps Environment
    Write-Host "  Exporting Container Apps Environment..." -ForegroundColor Cyan
    $env = az containerapp env show --name "nba-gbsv-model-env" --resource-group $SourceResourceGroup --output json 2>$null | ConvertFrom-Json
    if ($env) {
        $env | ConvertTo-Json -Depth 10 | Out-File (Join-Path $exportPath "container_app_env.json") -Encoding UTF8
        Write-Host "    ✓ Environment config exported" -ForegroundColor Green
    }

    # Export resource group tags
    Write-Host "  Exporting Resource Group configuration..." -ForegroundColor Cyan
    $rg = az group show --name $SourceResourceGroup --output json | ConvertFrom-Json
    $rg | ConvertTo-Json -Depth 10 | Out-File (Join-Path $exportPath "resource_group.json") -Encoding UTF8
    Write-Host "    ✓ Resource Group config exported" -ForegroundColor Green

    # Create summary document
    $summary = @"
# Azure NBA Configuration Export
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

## Source Resources
- Resource Group: $SourceResourceGroup
- Container App: nba-gbsv-api
- Container Registry: nbagbsacr
- Key Vault: nbagbs-keyvault
- Container Apps Environment: nba-gbsv-model-env

## Exported Files
- container_app.json - Container App full configuration
- acr.json - Container Registry configuration
- keyvault_secrets.json - Key Vault secret names (values not exported)
- container_app_env.json - Container Apps Environment configuration
- resource_group.json - Resource Group configuration

## Next Steps
1. Review exported configurations
2. Use infra/nba/create-nba-gbsv-model-rg.ps1 to deploy to new resource group
3. Or use infra/nba/deploy.ps1 to update existing deployment
"@
    $summary | Out-File (Join-Path $exportPath "README.md") -Encoding UTF8

    Write-Host ""
    Write-Host "  ✓ Configuration exported to: $exportPath" -ForegroundColor Green
    Write-Host ""
}

# Deploy to new environment
if ($Deploy) {
    Write-Host "[3/5] Deploying to Azure..." -ForegroundColor Yellow
    Write-Host "  Target Resource Group: $TargetResourceGroup" -ForegroundColor Cyan
    Write-Host "  Location: $Location" -ForegroundColor Cyan
    Write-Host ""

    # Check if target resource group exists
    $targetRgExists = az group show --name $TargetResourceGroup 2>$null
    if (-not $targetRgExists) {
        Write-Host "  Creating target resource group..." -ForegroundColor Cyan
        az group create --name $TargetResourceGroup --location $Location --output none
        Write-Host "    ✓ Resource group created" -ForegroundColor Green
    } else {
        Write-Host "  Target resource group already exists" -ForegroundColor Yellow
        $confirm = Read-Host "  Continue with deployment? (y/N)"
        if ($confirm -ne "y" -and $confirm -ne "Y") {
            Write-Host "  Deployment cancelled" -ForegroundColor Yellow
            exit 0
        }
    }

    # Run the provisioning script
    Write-Host ""
    Write-Host "  Running provisioning script..." -ForegroundColor Cyan
    $provisionScript = Join-Path $PSScriptRoot "..\infra\nba\create-nba-gbsv-model-rg.ps1"
    if (Test-Path $provisionScript) {
        & $provisionScript -ResourceGroup $TargetResourceGroup -Location $Location
    } else {
        Write-Host "  WARNING: Provisioning script not found at $provisionScript" -ForegroundColor Red
        Write-Host "  Please run manually: infra/nba/create-nba-gbsv-model-rg.ps1" -ForegroundColor Yellow
    }
}

# Create backup
if ($Backup) {
    Write-Host "[4/5] Creating backup..." -ForegroundColor Yellow
    $backupPath = Join-Path $exportPath "backup"
    if (-not (Test-Path $backupPath)) {
        New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
    }

    # Backup infrastructure files
    Write-Host "  Backing up infrastructure files..." -ForegroundColor Cyan
    $infraFiles = @(
        "infra/nba/main.bicep",
        "infra/nba/deploy.ps1",
        "infra/nba/provision.ps1",
        "infra/nba/create-nba-gbsv-model-rg.ps1",
        ".github/workflows/build-push-acr.yml",
        ".github/workflows/deploy-aca.yml",
        ".github/workflows/gbs-nba-deploy.yml"
    )

    foreach ($file in $infraFiles) {
        $sourceFile = Join-Path $PSScriptRoot "..\$file"
        if (Test-Path $sourceFile) {
            $destFile = Join-Path $backupPath (Split-Path $file -Leaf)
            Copy-Item $sourceFile $destFile -Force
            Write-Host "    ✓ Backed up: $file" -ForegroundColor Green
        }
    }

    Write-Host "  ✓ Backup created at: $backupPath" -ForegroundColor Green
}

# Summary
Write-Host ""
Write-Host "[5/5] Summary" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Green
if ($Backup -or -not $Deploy) {
    Write-Host "  Configuration exported to: $exportPath" -ForegroundColor Cyan
}
if ($Deploy) {
    Write-Host "  Deployment target: $TargetResourceGroup" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  To verify deployment:" -ForegroundColor Yellow
    Write-Host "    az containerapp show -n nba-gbsv-api -g $TargetResourceGroup --query properties.configuration.ingress.fqdn -o tsv" -ForegroundColor White
}
Write-Host ""
Write-Host "  Quick commands:" -ForegroundColor Yellow
Write-Host "    # View logs" -ForegroundColor White
Write-Host "    az containerapp logs show -n nba-gbsv-api -g $SourceResourceGroup --follow" -ForegroundColor Gray
Write-Host ""
Write-Host "    # Get API URL" -ForegroundColor White
Write-Host "    az containerapp show -n nba-gbsv-api -g $SourceResourceGroup --query properties.configuration.ingress.fqdn -o tsv" -ForegroundColor Gray
Write-Host ""
Write-Host "    # Deploy new version" -ForegroundColor White
Write-Host "    pwsh infra/nba/deploy.ps1 -Tag v6.10" -ForegroundColor Gray
Write-Host ""

