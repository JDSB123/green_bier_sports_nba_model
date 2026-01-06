<#
.SYNOPSIS
    Deploy NBA Container App infrastructure via Bicep.

.DESCRIPTION
    Wrapper script for deploying infra/nba/main.bicep to Azure.
    Reads version from VERSION file and prompts for required secrets.

.PARAMETER ResourceGroup
    Azure Resource Group name. Default: nba-gbsv-model-rg

.PARAMETER Tag
    Image/version tag override. Defaults to VERSION file content.

.PARAMETER WhatIf
    Run deployment in what-if mode (no changes applied).

.EXAMPLE
    .\deploy.ps1
    .\deploy.ps1 -Tag NBA_v33.0.11.0
    .\deploy.ps1 -WhatIf
#>

param(
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$Tag = "",
    [switch]$WhatIf
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

if (-not $TheOddsApiKey) {
    $TheOddsApiKey = Read-Host "Enter THE_ODDS_API_KEY" -AsSecureString | ConvertFrom-SecureString -AsPlainText
}
if (-not $ApiBasketballKey) {
    $ApiBasketballKey = Read-Host "Enter API_BASKETBALL_KEY" -AsSecureString | ConvertFrom-SecureString -AsPlainText
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
$BicepFile = Join-Path $ScriptDir "main.bicep"
$DeploymentName = "nba-deploy-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$params = @(
    "deployment", "group", "create",
    "--resource-group", $ResourceGroup,
    "--template-file", $BicepFile,
    "--name", $DeploymentName,
    "--parameters", "imageTag=$Tag",
    "--parameters", "theOddsApiKey=$TheOddsApiKey",
    "--parameters", "apiBasketballKey=$ApiBasketballKey"
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

