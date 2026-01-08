#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Store backtest model artifacts to Azure Blob Storage and local repository.
    
.DESCRIPTION
    Archives backtest model artifacts (models, results, configurations) to both
    the local nba-historical-data repository and Azure Blob Storage with
    versioning and timestamps.
    
.PARAMETER Version
    Model version (e.g., NBA_v33.0.11.0). Required.
    
.PARAMETER BacktestDate
    Date when backtest was run (YYYY-MM-DD format). Defaults to today.
    
.PARAMETER ModelPath
    Path to model file(s) or directory. Defaults to 'models/production'.
    
.PARAMETER ResultsPath
    Path to backtest results. Defaults to 'backtest_results'.
    
.PARAMETER StorageAccountName
    Azure storage account name. Defaults to 'nbagbsvstrg'.
    
.PARAMETER ResourceGroup
    Azure resource group name. Defaults to 'nba-gbsv-model-rg'.
    
.PARAMETER ContainerName
    Blob container name. Defaults to 'nbahistoricaldata'.
    
.PARAMETER HistoricalRepoPath
    Path to nba-historical-data repository. Defaults to '../nba-historical-data'.
    
.EXAMPLE
    .\scripts\store_backtest_model.ps1 -Version "NBA_v33.0.11.0" -BacktestDate "2025-01-15"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Version,
    
    [string]$BacktestDate = (Get-Date -Format "yyyy-MM-dd"),
    [string]$ModelPath = "models/production",
    [string]$ResultsPath = "backtest_results",
    [string]$StorageAccountName = "nbagbsvstrg",
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$ContainerName = "nbahistoricaldata",
    [string]$HistoricalRepoPath = "../nba-historical-data"
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== Storing Backtest Model ===" -ForegroundColor Cyan
Write-Host "Version: $Version" -ForegroundColor Gray
Write-Host "Backtest Date: $BacktestDate" -ForegroundColor Gray
Write-Host "Container: $ContainerName`n" -ForegroundColor Gray

# Validate date format
try {
    $dateObj = [DateTime]::ParseExact($BacktestDate, "yyyy-MM-dd", $null)
} catch {
    Write-Error "Invalid date format. Use YYYY-MM-DD format."
    exit 1
}

# Generate timestamp and safe version string
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$versionSafe = $Version -replace '[^a-zA-Z0-9._-]', '_'

# Create local archive structure
$localBacktestDir = Join-Path $HistoricalRepoPath "models/backtest/$versionSafe/$BacktestDate"
New-Item -ItemType Directory -Path $localBacktestDir -Force | Out-Null
Write-Host "✓ Created local archive directory: $localBacktestDir" -ForegroundColor Green

# Copy model files
if (Test-Path $ModelPath) {
    Write-Host "`nCopying model files..." -ForegroundColor Yellow
    $modelDest = Join-Path $localBacktestDir "models"
    New-Item -ItemType Directory -Path $modelDest -Force | Out-Null
    
    if ((Get-Item $ModelPath).PSIsContainer) {
        Copy-Item -Path "$ModelPath/*" -Destination $modelDest -Recurse -Force
    } else {
        Copy-Item -Path $ModelPath -Destination $modelDest -Force
    }
    Write-Host "✓ Model files copied" -ForegroundColor Green
} else {
    Write-Warning "Model path not found: $ModelPath"
}

# Copy backtest results
if (Test-Path $ResultsPath) {
    Write-Host "`nCopying backtest results..." -ForegroundColor Yellow
    $resultsDest = Join-Path $localBacktestDir "results"
    New-Item -ItemType Directory -Path $resultsDest -Force | Out-Null
    
    Copy-Item -Path "$ResultsPath/*" -Destination $resultsDest -Recurse -Force
    Write-Host "✓ Backtest results copied" -ForegroundColor Green
} else {
    Write-Warning "Results path not found: $ResultsPath"
}

# Create metadata file
$metadata = @{
    model_version = $Version
    backtest_date = $BacktestDate
    timestamp = $timestamp
    stored_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    model_path = $ModelPath
    results_path = $ResultsPath
} | ConvertTo-Json

$metadataPath = Join-Path $localBacktestDir "metadata.json"
Set-Content -Path $metadataPath -Value $metadata -Encoding UTF8
Write-Host "✓ Created metadata file" -ForegroundColor Green

# Upload to Azure Blob Storage
Write-Host "`nUploading to Azure Blob Storage..." -ForegroundColor Yellow

try {
    # Get storage account key
    $storageKey = az storage account keys list `
        --account-name $StorageAccountName `
        --resource-group $ResourceGroup `
        --query "[0].value" `
        --output tsv
    
    if (-not $storageKey) {
        throw "Failed to retrieve storage account key"
    }
    
    # Upload entire directory
    $blobPrefix = "models/backtest/$versionSafe/$BacktestDate"
    $files = Get-ChildItem -Path $localBacktestDir -Recurse -File
    
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring((Resolve-Path $localBacktestDir).Path.Length + 1)
        $blobPath = "$blobPrefix/$relativePath".Replace('\', '/')
        
        az storage blob upload `
            --account-name $StorageAccountName `
            --account-key $storageKey `
            --container-name $ContainerName `
            --name $blobPath `
            --file $file.FullName `
            --overwrite `
            | Out-Null
        
        Write-Host "  ✓ $blobPath" -ForegroundColor Gray
    }
    
    Write-Host "✓ Successfully uploaded to Azure Blob Storage" -ForegroundColor Green
    
} catch {
    Write-Warning "Failed to upload to Azure: $_"
    Write-Host "Local archive preserved at: $localBacktestDir" -ForegroundColor Yellow
}

Write-Host "`n=== Storage Complete ===" -ForegroundColor Cyan
Write-Host "Local Archive: $localBacktestDir" -ForegroundColor Green
Write-Host "Azure Blob Prefix: models/backtest/$versionSafe/$BacktestDate" -ForegroundColor Green
Write-Host "`n"
