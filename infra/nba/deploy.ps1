<#
.SYNOPSIS
    Deploy NBA Container App infrastructure via Bicep.

.DESCRIPTION
    Wrapper script for deploying infra/nba/prediction.bicep to Azure.
    Reads version from VERSION file and prompts for required secrets.

.PARAMETER ResourceGroup
    Azure Resource Group name. Default: nba-gbsv-model-rg

.PARAMETER Tag
    Image/version tag override. Defaults to VERSION file content.

.PARAMETER WhatIf
    Run deployment in what-if mode (no changes applied).

.PARAMETER SkipReadiness
    Skip the live production readiness checks before deploy.

.EXAMPLE
    .\deploy.ps1
    .\deploy.ps1 -Tag (Get-Content VERSION -Raw).Trim()
    .\deploy.ps1 -WhatIf
#>

param(
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$Tag = "",
    [switch]$WhatIf,
    [switch]$SkipReadiness
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)

# Read version from VERSION file if not provided
if (-not $Tag) {
    $VersionFile = Join-Path $RepoRoot "VERSION"
    if (Test-Path $VersionFile) {
        $Tag = (Get-Content $VersionFile -Raw).Trim()
        Write-Host "Using version from VERSION file: $Tag" -ForegroundColor Cyan
    } else {
        Write-Error "VERSION file not found and -Tag not provided"
        exit 1
    }
}

# Check for required environment variables or prompt
$TheOddsApiKey = $env:THE_ODDS_API_KEY
$ApiBasketballKey = $env:API_BASKETBALL_KEY
$ActionNetworkUser = $env:ACTION_NETWORK_USERNAME
$ActionNetworkPass = $env:ACTION_NETWORK_PASSWORD

if (-not $TheOddsApiKey) {
    $TheOddsApiKey = Read-Host "Enter THE_ODDS_API_KEY" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}
if (-not $ApiBasketballKey) {
    $ApiBasketballKey = Read-Host "Enter API_BASKETBALL_KEY" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}
if (-not $ActionNetworkUser) {
    $ActionNetworkUser = Read-Host "Enter ACTION_NETWORK_USERNAME (required for strict splits)"
}
if (-not $ActionNetworkPass) {
    $ActionNetworkPass = Read-Host "Enter ACTION_NETWORK_PASSWORD (required for strict splits)" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}

# Export env vars for readiness script
$env:THE_ODDS_API_KEY = $TheOddsApiKey
$env:API_BASKETBALL_KEY = $ApiBasketballKey
$env:ACTION_NETWORK_USERNAME = $ActionNetworkUser
$env:ACTION_NETWORK_PASSWORD = $ActionNetworkPass
$env:PREDICTION_FEATURE_MODE = "strict"
$env:MIN_FEATURE_COMPLETENESS = "0.95"
$env:REQUIRE_ACTION_NETWORK_SPLITS = "true"
$env:REQUIRE_REAL_SPLITS = "true"
$env:REQUIRE_SHARP_BOOK_DATA = "true"
$env:REQUIRE_INJURY_FETCH_SUCCESS = "true"

if (-not $SkipReadiness) {
    Write-Host "`nRunning production readiness checks (live)..." -ForegroundColor Yellow
    & python (Join-Path $RepoRoot "scripts" "predict_validate_production_readiness.py") --live
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Production readiness checks failed. Fix issues before deploy."
        exit $LASTEXITCODE
    }
}

# Validate Azure CLI login
Write-Host "Checking Azure CLI login..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Error "Not logged into Azure CLI. Run 'az login' first."
    exit 1
}
Write-Host "Logged in as: $($account.user.name) | Subscription: $($account.name)" -ForegroundColor Green

# Validate resource group exists
$rgExists = az group exists --name $ResourceGroup 2>$null
if ($rgExists -ne "true") {
    Write-Error "Resource group '$ResourceGroup' does not exist"
    exit 1
}

# Build deployment command
$BicepFile = Join-Path $ScriptDir "prediction.bicep"
$DeploymentName = "nba-deploy-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$params = @(
    "deployment", "group", "create",
    "--resource-group", $ResourceGroup,
    "--template-file", $BicepFile,
    "--name", $DeploymentName,
    "--parameters", "imageTag=$Tag",
    "--parameters", "theOddsApiKey=$TheOddsApiKey",
    "--parameters", "apiBasketballKey=$ApiBasketballKey",
    "--parameters", "actionNetworkUsername=$ActionNetworkUser",
    "--parameters", "actionNetworkPassword=$ActionNetworkPass"
)

if ($WhatIf) {
    $params[2] = "what-if"
    Write-Host "`n=== WHAT-IF MODE (no changes will be applied) ===" -ForegroundColor Yellow
}

Write-Host "`nDeploying to '$ResourceGroup' with tag '$Tag'..." -ForegroundColor Cyan
Write-Host "Bicep file: $BicepFile" -ForegroundColor Gray

# Execute deployment
& az @params

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Deployment successful!" -ForegroundColor Green

    if (-not $WhatIf) {
        # Show outputs
        Write-Host "`nDeployment outputs:" -ForegroundColor Cyan
        az deployment group show `
            --resource-group $ResourceGroup `
            --name $DeploymentName `
            --query properties.outputs
    }
} else {
    Write-Error "Deployment failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
