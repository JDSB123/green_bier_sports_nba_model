#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Verify historical data storage setup is complete and ready.
    
.DESCRIPTION
    Checks that all components of the historical data storage system are
    properly configured and ready to use.
    
.EXAMPLE
    .\scripts\verify_historical_data_setup.ps1
#>

$ErrorActionPreference = "Continue"

Write-Host "`n=== Verifying Historical Data Storage Setup ===" -ForegroundColor Cyan
Write-Host ""

$allChecksPassed = $true

# Check 1: Azure Infrastructure
Write-Host "[1/6] Checking Azure Infrastructure..." -ForegroundColor Yellow
try {
    $containers = az storage container list `
        --account-name nbagbsvstrg `
        --resource-group nba-gbsv-model-rg `
        --output json 2>$null | ConvertFrom-Json
    
    if ($containers) {
        $nbahistoricaldata = $containers | Where-Object { $_.name -eq "nbahistoricaldata" }
        if ($nbahistoricaldata) {
            Write-Host "  ✓ Container nbahistoricaldata exists" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Container nbahistoricaldata not found" -ForegroundColor Red
            Write-Host "    Run: cd infra/nba; az deployment group create ..." -ForegroundColor Yellow
            $allChecksPassed = $false
        }
    } else {
        Write-Host "  ✗ Could not list containers (check Azure CLI login)" -ForegroundColor Red
        $allChecksPassed = $false
    }
} catch {
    Write-Host "  ✗ Azure CLI error: $_" -ForegroundColor Red
    Write-Host "    Ensure Azure CLI is installed and you are logged in" -ForegroundColor Yellow
    $allChecksPassed = $false
}

# Check 2: Scripts exist
Write-Host "`n[2/6] Checking scripts..." -ForegroundColor Yellow
$scripts = @(
    "scripts/init_historical_data_repo.ps1",
    "scripts/archive_picks_to_azure.ps1",
    "scripts/sync_historical_data_to_azure.ps1",
    "scripts/store_backtest_model.ps1"
)

foreach ($script in $scripts) {
    if (Test-Path $script) {
        Write-Host "  ✓ $script" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $script not found" -ForegroundColor Red
        $allChecksPassed = $false
    }
}

# Check 3: Bicep file updated
Write-Host "`n[3/6] Checking Bicep configuration..." -ForegroundColor Yellow
$bicepFile = "infra/nba/main.bicep"
if (Test-Path $bicepFile) {
    $bicepContent = Get-Content $bicepFile -Raw
    if ($bicepContent -match "nbahistoricaldata") {
        Write-Host "  ✓ Bicep file includes nbahistoricaldata container" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Bicep file missing nbahistoricaldata container" -ForegroundColor Red
        $allChecksPassed = $false
    }
} else {
    Write-Host "  ✗ Bicep file not found: $bicepFile" -ForegroundColor Red
    $allChecksPassed = $false
}

# Check 4: Documentation exists
Write-Host "`n[4/6] Checking documentation..." -ForegroundColor Yellow
$docs = @(
    "docs/HISTORICAL_DATA_STORAGE.md",
    "docs/HISTORICAL_DATA_SETUP_SUMMARY.md"
)

foreach ($doc in $docs) {
    if (Test-Path $doc) {
        Write-Host "  ✓ $doc" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $doc not found" -ForegroundColor Red
        $allChecksPassed = $false
    }
}

# Check 5: Local directories structure
Write-Host "`n[5/6] Checking local directory structure..." -ForegroundColor Yellow
$localDirs = @(
    "data/archived_picks",
    "data/historical"
)

foreach ($dir in $localDirs) {
    if (Test-Path $dir) {
        Write-Host "  ✓ $dir exists" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $dir does not exist (will be created when needed)" -ForegroundColor Yellow
    }
}

# Check 6: Azure CLI availability
Write-Host "`n[6/6] Checking Azure CLI..." -ForegroundColor Yellow
try {
    $azVersion = az version --output json 2>$null | ConvertFrom-Json
    if ($azVersion) {
        Write-Host "  ✓ Azure CLI is installed" -ForegroundColor Green
        $azCliVersion = $azVersion["azure-cli"]
        Write-Host "    Version: $azCliVersion" -ForegroundColor Gray
        
        # Check login status
        $account = az account show --output json 2>$null | ConvertFrom-Json
        if ($account) {
            Write-Host "  ✓ Logged in as: $($account.user.name)" -ForegroundColor Green
            Write-Host "    Subscription: $($account.name)" -ForegroundColor Gray
        } else {
            Write-Host "  ✗ Not logged in to Azure" -ForegroundColor Red
            Write-Host "    Run: az login" -ForegroundColor Yellow
            $allChecksPassed = $false
        }
    } else {
        Write-Host "  ✗ Azure CLI not found" -ForegroundColor Red
        Write-Host "    Install: winget install -e --id Microsoft.AzureCLI" -ForegroundColor Yellow
        $allChecksPassed = $false
    }
} catch {
    Write-Host "  ✗ Azure CLI check failed: $_" -ForegroundColor Red
    $allChecksPassed = $false
}

# Summary
Write-Host "`n=== Verification Summary ===" -ForegroundColor Cyan
if ($allChecksPassed) {
    Write-Host "✓ All checks passed! Historical data storage is ready." -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Initialize repository: .\scripts\init_historical_data_repo.ps1" -ForegroundColor Gray
    Write-Host "  2. Archive picks: .\scripts\archive_picks_to_azure.ps1 -Date YYYY-MM-DD" -ForegroundColor Gray
    Write-Host "  3. Sync historical data: .\scripts\sync_historical_data_to_azure.ps1" -ForegroundColor Gray
} else {
    Write-Host "✗ Some checks failed. Please fix the issues above." -ForegroundColor Red
    Write-Host "`nCommon fixes:" -ForegroundColor Yellow
    Write-Host "  - Deploy infrastructure: cd infra/nba; az deployment group create ..." -ForegroundColor Gray
    Write-Host "  - Login to Azure: az login" -ForegroundColor Gray
    Write-Host "  - Install Azure CLI: winget install -e --id Microsoft.AzureCLI" -ForegroundColor Gray
}

Write-Host "`n"
