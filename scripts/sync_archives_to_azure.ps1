# Sync archives to Azure Blob Storage
# SINGLE SOURCE OF TRUTH: Archives stored in Azure, not git
#
# Usage:
#   .\scripts\sync_archives_to_azure.ps1              # Sync all archives
#   .\scripts\sync_archives_to_azure.ps1 -Cleanup     # Sync and delete local

param(
    [switch]$Cleanup,
    [string]$StorageAccount = "nbagbsvstrg",
    [string]$Container = "nbahistoricaldata"
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SYNC ARCHIVES TO AZURE BLOB STORAGE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Ensure logged in
Write-Host "`nChecking Azure login..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Not logged in. Running az login..." -ForegroundColor Red
    az login
}
Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green

# Archives to sync
$archives = @(
    @{ Local = "scripts/archive"; Remote = "archives/scripts" },
    @{ Local = "archive/picks"; Remote = "archives/picks" },
    @{ Local = "archive/predictions"; Remote = "archives/predictions" },
    @{ Local = "archive/slate_outputs"; Remote = "archives/slate_outputs" },
    @{ Local = "archive/odds_snapshots"; Remote = "archives/odds_snapshots" },
    @{ Local = "archive/analysis"; Remote = "archives/analysis" }
)

# External data (too large for git)
$externalData = @(
    @{ Local = "data/external/nba_database"; Remote = "external/nba_database" },
    @{ Local = "data/external/kaggle"; Remote = "external/kaggle" },
    @{ Local = "data/external/fivethirtyeight"; Remote = "external/fivethirtyeight" }
)

Write-Host "`n[1/3] Syncing archive folders..." -ForegroundColor Yellow
foreach ($arch in $archives) {
    $localPath = Join-Path $PSScriptRoot ".." $arch.Local
    if (Test-Path $localPath) {
        $fileCount = (Get-ChildItem $localPath -Recurse -File).Count
        Write-Host "  Uploading $($arch.Local) ($fileCount files) -> $($arch.Remote)" -ForegroundColor Gray
        az storage blob upload-batch `
            --account-name $StorageAccount `
            --destination $Container `
            --destination-path $arch.Remote `
            --source $localPath `
            --overwrite `
            2>&1 | Out-Null
        Write-Host "  [OK] $($arch.Local)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $($arch.Local) (not found)" -ForegroundColor DarkGray
    }
}

Write-Host "`n[2/3] Syncing large external data..." -ForegroundColor Yellow
foreach ($ext in $externalData) {
    $localPath = Join-Path $PSScriptRoot ".." $ext.Local
    if (Test-Path $localPath) {
        $sizeMB = [math]::Round((Get-ChildItem $localPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
        Write-Host "  Uploading $($ext.Local) ($sizeMB MB) -> $($ext.Remote)" -ForegroundColor Gray
        az storage blob upload-batch `
            --account-name $StorageAccount `
            --destination $Container `
            --destination-path $ext.Remote `
            --source $localPath `
            --overwrite `
            2>&1 | Out-Null
        Write-Host "  [OK] $($ext.Local)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $($ext.Local) (not found)" -ForegroundColor DarkGray
    }
}

if ($Cleanup) {
    Write-Host "`n[3/3] Cleaning up local archive copies..." -ForegroundColor Yellow
    foreach ($arch in $archives) {
        $localPath = Join-Path $PSScriptRoot ".." $arch.Local
        if (Test-Path $localPath) {
            Write-Host "  Removing $($arch.Local)..." -ForegroundColor Gray
            Remove-Item $localPath -Recurse -Force
            Write-Host "  [OK] Deleted $($arch.Local)" -ForegroundColor Green
        }
    }
} else {
    Write-Host "`n[3/3] Skipping cleanup (use -Cleanup flag to delete local copies)" -ForegroundColor DarkGray
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "SYNC COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Archives synced to: $StorageAccount/$Container" -ForegroundColor White
Write-Host ""
Write-Host "To download archives later:" -ForegroundColor White
Write-Host "  az storage blob download-batch --account-name $StorageAccount --source $Container --destination . --pattern 'archives/*'" -ForegroundColor Gray
