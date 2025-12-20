# NBA Prediction System v5.0 - SINGLE ENTRY POINT (Production Picks)
# Single Source of Truth:
# - Manually initiated
# - Runs analysis for: today / tomorrow / a date
# - Optionally filters to: one game or multiple games (comma-separated)
#
# This script intentionally does NOT run backtests.

param(
    [string]$Date = "today",
    [string]$Matchup = ""
)

$ErrorActionPreference = "Stop"

Write-Host "NBA Prediction System v5.0 - Production Picks (Single Entry Point)" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is available
if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH."
    exit 1
}

# Check if Python is available
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH."
    exit 1
}

# Check if .env file exists (used by docker compose)
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Warning ".env file not found. Creating from .env.example if it exists..."
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" $envFile
        Write-Host "Created .env from .env.example - please fill in your API keys!" -ForegroundColor Yellow
    } else {
        Write-Error ".env file is required. Please create one with your API keys."
        exit 1
    }
}

Write-Host "Delegating to the single source-of-truth runner..." -ForegroundColor Yellow
Write-Host ""

# Build args for scripts/run_slate.py
$analysisArgs = @()
if ($Date) {
    $analysisArgs += "--date"
    $analysisArgs += $Date
}
if ($Matchup) {
    $analysisArgs += "--matchup"
    $analysisArgs += $Matchup
}

Write-Host "Running analysis..." -ForegroundColor Cyan
Write-Host ""
python scripts/run_slate.py $analysisArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Analysis failed. Check logs: docker compose logs strict-api" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[OK] Analysis complete. Reports saved to data/processed/" -ForegroundColor Green
