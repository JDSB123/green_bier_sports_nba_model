#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Archive picks to Azure Blob Storage before front-end overwrites.
    
.DESCRIPTION
    Archives picks with unique timestamps and versions to the 'nbahistoricaldata'
    container in Azure Blob Storage. This ensures picks are preserved before
    being overwritten on any front ends.
    
.PARAMETER Date
    Date of picks to archive (YYYY-MM-DD format). Defaults to today.
    
.PARAMETER ModelVersion
    Model version (e.g., NBA_v33.0.11.0). Defaults to NBA_MODEL_VERSION env var.
    
.PARAMETER StorageAccountName
    Azure storage account name. Defaults to 'nbagbsvstrg'.
    
.PARAMETER ResourceGroup
    Azure resource group name. Defaults to 'nba-gbsv-model-rg'.
    
.PARAMETER ContainerName
    Blob container name. Defaults to 'nbahistoricaldata'.
    
.PARAMETER PicksFile
    Path to picks JSON file. Defaults to data/processed/tracking/live_picks.jsonl
    
.EXAMPLE
    .\scripts\archive_picks_to_azure.ps1 -Date "2025-01-15"
    
.EXAMPLE
    .\scripts\archive_picks_to_azure.ps1 -Date "2025-01-15" -ModelVersion "NBA_v33.0.11.0"
#>

param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [string]$ModelVersion = $env:NBA_MODEL_VERSION,
    [string]$StorageAccountName = "nbagbsvstrg",
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$ContainerName = "nbahistoricaldata",
    [string]$PicksFile = "data/processed/tracking/live_picks.jsonl"
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== Archiving Picks to Azure Blob Storage ===" -ForegroundColor Cyan
Write-Host "Date: $Date" -ForegroundColor Gray
Write-Host "Model Version: $ModelVersion" -ForegroundColor Gray
Write-Host "Container: $ContainerName`n" -ForegroundColor Gray

# Validate date format
try {
    [DateTime]::ParseExact($Date, "yyyy-MM-dd", $null) | Out-Null
} catch {
    Write-Error "Invalid date format. Use YYYY-MM-DD format."
    exit 1
}

# Check if picks file exists
if (-not (Test-Path $PicksFile)) {
    Write-Warning "Picks file not found: $PicksFile"
    Write-Host "Attempting to find picks from prediction logs..." -ForegroundColor Yellow
    
    $predictionLogsDir = "data/processed/prediction_logs"
    $logFile = Join-Path $predictionLogsDir "predictions_$Date.jsonl"
    
    if (Test-Path $logFile) {
        $PicksFile = $logFile
        Write-Host "Using prediction log: $logFile" -ForegroundColor Green
    } else {
        Write-Error "No picks file found for date $Date"
        exit 1
    }
}

# Generate unique timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$versionSafe = $ModelVersion -replace '[^a-zA-Z0-9._-]', '_'

# Create blob path with version and timestamp
$blobPath = "archived_picks/$Date/v${versionSafe}_${timestamp}.jsonl"
$localArchivePath = "data/archived_picks/by_date/$Date"
New-Item -ItemType Directory -Path $localArchivePath -Force | Out-Null

# Copy file locally first
$localArchiveFile = Join-Path $localArchivePath "picks_${timestamp}.jsonl"
Copy-Item $PicksFile $localArchiveFile -Force
Write-Host "✓ Copied picks to local archive: $localArchiveFile" -ForegroundColor Green

# Upload to Azure Blob Storage
Write-Host "`nUploading to Azure Blob Storage..." -ForegroundColor Yellow

try {
    # Check if Azure CLI is available
    $azVersion = az version --output json 2>$null | ConvertFrom-Json
    if (-not $azVersion) {
        throw "Azure CLI not found"
    }
    
    # Get storage account key
    Write-Host "  Retrieving storage account key..." -ForegroundColor Gray
    $storageKey = az storage account keys list `
        --account-name $StorageAccountName `
        --resource-group $ResourceGroup `
        --query "[0].value" `
        --output tsv
    
    if (-not $storageKey) {
        throw "Failed to retrieve storage account key"
    }
    
    # Upload blob
    Write-Host "  Uploading blob: $blobPath" -ForegroundColor Gray
    az storage blob upload `
        --account-name $StorageAccountName `
        --account-key $storageKey `
        --container-name $ContainerName `
        --name $blobPath `
        --file $PicksFile `
        --overwrite `
        | Out-Null
    
    Write-Host "✓ Successfully uploaded to Azure Blob Storage" -ForegroundColor Green
    Write-Host "  Container: $ContainerName" -ForegroundColor Gray
    Write-Host "  Blob: $blobPath" -ForegroundColor Gray
    
    # Also create a versioned symlink/copy
    $versionPath = "archived_picks/by_version/$versionSafe/$Date/picks_${timestamp}.jsonl"
    Write-Host "  Creating versioned copy..." -ForegroundColor Gray
    
    az storage blob upload `
        --account-name $StorageAccountName `
        --account-key $storageKey `
        --container-name $ContainerName `
        --name $versionPath `
        --file $PicksFile `
        --overwrite `
        | Out-Null
    
    Write-Host "✓ Created versioned copy: $versionPath" -ForegroundColor Green
    
    # Create metadata file
    $metadata = @{
        date = $Date
        timestamp = $timestamp
        model_version = $ModelVersion
        blob_path = $blobPath
        version_path = $versionPath
        archived_at = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        source_file = $PicksFile
    } | ConvertTo-Json
    
    $metadataBlobPath = "archived_picks/$Date/v${versionSafe}_${timestamp}_metadata.json"
    $metadataFile = [System.IO.Path]::GetTempFileName()
    Set-Content -Path $metadataFile -Value $metadata -Encoding UTF8
    
    az storage blob upload `
        --account-name $StorageAccountName `
        --account-key $storageKey `
        --container-name $ContainerName `
        --name $metadataBlobPath `
        --file $metadataFile `
        --overwrite `
        | Out-Null
    
    Remove-Item $metadataFile -Force
    Write-Host "✓ Created metadata file: $metadataBlobPath" -ForegroundColor Green
    
} catch {
    Write-Error "Failed to upload to Azure: $_"
    Write-Host "`nLocal archive preserved at: $localArchiveFile" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Archive Complete ===" -ForegroundColor Cyan
Write-Host "Blob Storage Path: $blobPath" -ForegroundColor Green
Write-Host "Local Archive: $localArchiveFile" -ForegroundColor Green
Write-Host "`n"
