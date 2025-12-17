# NBA Basketball Prediction System v5.0 BETA - Setup Script (PowerShell)
# Run this script to set up the development environment

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NBA Basketball v5.0 BETA Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
$projectDir = "C:\Users\JB\green-bier-ventures\nba_v5.0_BETA"
if ((Get-Location).Path -ne $projectDir) {
    Write-Host "Changing to project directory..." -ForegroundColor Yellow
    Set-Location $projectDir
}

# Check for Docker
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "Docker found" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker not found. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check for .env file
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "IMPORTANT: Edit .env and add your API keys!" -ForegroundColor Red
        Write-Host "  - THE_ODDS_API_KEY" -ForegroundColor Yellow
        Write-Host "  - API_BASKETBALL_KEY" -ForegroundColor Yellow
    } else {
        Write-Host "Creating basic .env file..." -ForegroundColor Yellow
        @"
THE_ODDS_API_KEY=
API_BASKETBALL_KEY=
DB_PASSWORD=nba_dev_password
"@ | Out-File -FilePath ".env" -Encoding utf8
        Write-Host "IMPORTANT: Edit .env and add your API keys!" -ForegroundColor Red
    }
} else {
    Write-Host ".env file already exists" -ForegroundColor Green
}

# Create Python venv for local development (optional)
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment (optional, for local dev)..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "To activate: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit .env and add your API keys" -ForegroundColor Yellow
Write-Host "  2. Start services: docker-compose up -d" -ForegroundColor Yellow
Write-Host "  3. Check health: curl http://localhost:8080/health" -ForegroundColor Yellow
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f prediction-service" -ForegroundColor Yellow
Write-Host "  docker-compose logs -f odds-ingestion" -ForegroundColor Yellow
Write-Host ""
