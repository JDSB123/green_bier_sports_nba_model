#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Sync historical data ingestion to Azure Blob Storage.
    
.DESCRIPTION
    Syncs historical data from local data/historical/ directory to Azure Blob Storage
    container 'nbahistoricaldata'. Data is organized by season and type with
    unique timestamps to prevent overwrites.
    
.PARAMETER Season
    Season to sync (e.g., 2024-2025). If not specified, syncs all seasons.
    
.PARAMETER DataType
    Type of data to sync: 'events', 'odds', 'period_odds', 'player_props', 'exports', or 'all'.
    Defaults to 'all'.
    
.PARAMETER StorageAccountName
    Azure storage account name. Defaults to 'nbagbsvstrg'.
    
.PARAMETER ResourceGroup
    Azure resource group name. Defaults to 'nba-gbsv-model-rg'.
    
.PARAMETER ContainerName
    Blob container name. Defaults to 'nbahistoricaldata'.
    
.PARAMETER LocalDataPath
    Local path to historical data. Defaults to 'data/historical'.
    
.PARAMETER DryRun
    Show what would be synced without actually uploading.
    
.EXAMPLE
    .\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"
    
.EXAMPLE
    .\scripts\sync_historical_data_to_azure.ps1 -DataType "odds" -DryRun
#>

param(
    [string]$Season = "",
    [ValidateSet("events", "odds", "period_odds", "player_props", "exports", "all")]
    [string]$DataType = "all",
    [string]$StorageAccountName = "nbagbsvstrg",
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$ContainerName = "nbahistoricaldata",
    [string]$LocalDataPath = "data/historical",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== Syncing Historical Data to Azure Blob Storage ===" -ForegroundColor Cyan
Write-Host "Season: $(if ($Season) { $Season } else { 'All seasons' })" -ForegroundColor Gray
Write-Host "Data Type: $DataType" -ForegroundColor Gray
Write-Host "Container: $ContainerName" -ForegroundColor Gray
Write-Host "Dry Run: $DryRun`n" -ForegroundColor Gray

# Validate local data path
if (-not (Test-Path $LocalDataPath)) {
    Write-Error "Local data path not found: $LocalDataPath"
    exit 1
}

# Get storage account key
if (-not $DryRun) {
    Write-Host "Retrieving storage account key..." -ForegroundColor Yellow
    try {
        $storageKey = az storage account keys list `
            --account-name $StorageAccountName `
            --resource-group $ResourceGroup `
            --query "[0].value" `
            --output tsv
        
        if (-not $storageKey) {
            throw "Failed to retrieve storage account key"
        }
        Write-Host "✓ Storage account key retrieved" -ForegroundColor Green
    } catch {
        Write-Error "Failed to connect to Azure: $_"
        exit 1
    }
}

# Function to sync files
function Sync-Files {
    param(
        [string]$LocalPath,
        [string]$BlobPrefix,
        [string]$Pattern = "*.json"
    )
    
    if (-not (Test-Path $LocalPath)) {
        Write-Warning "Path not found: $LocalPath"
        return
    }
    
    $files = Get-ChildItem -Path $LocalPath -Recurse -Filter $Pattern -File
    $count = 0
    
    foreach ($file in $files) {
        # Skip if season filter is specified
        if ($Season -and $file.FullName -notmatch $Season) {
            continue
        }
        
        # Calculate relative path
        $relativePath = $file.FullName.Substring((Resolve-Path $LocalDataPath).Path.Length + 1)
        $blobPath = "$BlobPrefix/$relativePath".Replace('\', '/')
        
        if ($DryRun) {
            Write-Host "  [DRY RUN] Would upload: $blobPath" -ForegroundColor Gray
        } else {
            try {
                az storage blob upload `
                    --account-name $StorageAccountName `
                    --account-key $storageKey `
                    --container-name $ContainerName `
                    --name $blobPath `
                    --file $file.FullName `
                    --overwrite `
                    | Out-Null
                
                Write-Host "  ✓ $blobPath" -ForegroundColor Green
            } catch {
                Write-Warning "  ✗ Failed to upload $blobPath : $_"
            }
        }
        $count++
    }
    
    return $count
}

# Sync based on data type
$totalFiles = 0

if ($DataType -eq "all" -or $DataType -eq "events") {
    Write-Host "`nSyncing events..." -ForegroundColor Yellow
    $eventsPath = Join-Path $LocalDataPath "the_odds/events"
    $count = Sync-Files -LocalPath $eventsPath -BlobPrefix "historical/the_odds/events"
    $totalFiles += $count
    Write-Host "  Processed $count event files" -ForegroundColor Gray
}

if ($DataType -eq "all" -or $DataType -eq "odds") {
    Write-Host "`nSyncing odds..." -ForegroundColor Yellow
    $oddsPath = Join-Path $LocalDataPath "the_odds/odds"
    $count = Sync-Files -LocalPath $oddsPath -BlobPrefix "historical/the_odds/odds"
    $totalFiles += $count
    Write-Host "  Processed $count odds files" -ForegroundColor Gray
}

if ($DataType -eq "all" -or $DataType -eq "period_odds") {
    Write-Host "`nSyncing period odds..." -ForegroundColor Yellow
    $periodOddsPath = Join-Path $LocalDataPath "the_odds/period_odds"
    $count = Sync-Files -LocalPath $periodOddsPath -BlobPrefix "historical/the_odds/period_odds"
    $totalFiles += $count
    Write-Host "  Processed $count period odds files" -ForegroundColor Gray
}

if ($DataType -eq "all" -or $DataType -eq "player_props") {
    Write-Host "`nSyncing player props..." -ForegroundColor Yellow
    $propsPath = Join-Path $LocalDataPath "the_odds/player_props"
    $count = Sync-Files -LocalPath $propsPath -BlobPrefix "historical/the_odds/player_props"
    $totalFiles += $count
    Write-Host "  Processed $count player props files" -ForegroundColor Gray
}

if ($DataType -eq "all" -or $DataType -eq "exports") {
    Write-Host "`nSyncing exports..." -ForegroundColor Yellow
    $exportsPath = Join-Path $LocalDataPath "exports"
    $count = Sync-Files -LocalPath $exportsPath -BlobPrefix "historical/exports" -Pattern "*.*"
    $totalFiles += $count
    Write-Host "  Processed $count export files" -ForegroundColor Gray
}

# Sync metadata
if ($DataType -eq "all") {
    Write-Host "`nSyncing metadata..." -ForegroundColor Yellow
    $metadataPath = Join-Path $LocalDataPath "the_odds/metadata"
    $count = Sync-Files -LocalPath $metadataPath -BlobPrefix "historical/the_odds/metadata"
    $totalFiles += $count
    Write-Host "  Processed $count metadata files" -ForegroundColor Gray
}

Write-Host "`n=== Sync Complete ===" -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "Dry run: Would sync $totalFiles files" -ForegroundColor Yellow
} else {
    Write-Host "Successfully synced $totalFiles files to Azure Blob Storage" -ForegroundColor Green
    Write-Host "Container: $ContainerName" -ForegroundColor Gray
}
Write-Host "`n"
