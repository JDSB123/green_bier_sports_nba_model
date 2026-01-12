# Runs a backtest using historical data pulled from Azure Blob Storage.
# Usage:
#   pwsh ./scripts/run_backtest_from_blob.ps1                  # full backtest (default)
#   pwsh ./scripts/run_backtest_from_blob.ps1 -Service backtest-only
param(
    [string]$AccountName = "nbagbsvstrg",
    [string]$ContainerName = "nbahistoricaldata",
    [string]$Prefix = "historical/*",
    [string]$TempDir = "data/historical",
    [string]$ComposeFile = "docker-compose.backtest.yml",
    [string]$Service = "backtest-full",
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"

Write-Host "=== Azure Blob -> local cache (temporary) ===" -ForegroundColor Cyan
if (Test-Path $TempDir) {
    Write-Host "Removing existing temp directory: $TempDir" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $TempDir
}

Write-Host "Downloading $Prefix from $ContainerName in account $AccountName ..." -ForegroundColor Cyan
az storage blob download-batch `
    --account-name $AccountName `
    --auth-mode login `
    --source $ContainerName `
    --destination $TempDir `
    --pattern $Prefix `
    --no-progress

Write-Host "=== Running docker compose service: $Service ===" -ForegroundColor Cyan
docker compose -f $ComposeFile up $Service
$exitCode = $LASTEXITCODE

if (-not $NoClean) {
    Write-Host "Cleaning temp directory: $TempDir" -ForegroundColor Yellow
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir
    }
}

if ($exitCode -ne 0) {
    Write-Error "Backtest run failed with exit code $exitCode"
    exit $exitCode
}

Write-Host "Backtest completed successfully." -ForegroundColor Green
