Param(
    [switch]$Compose,
    [switch]$ComposeDown,
    [switch]$API,
    [switch]$ManualFetch,
    [switch]$Worker
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root

function Ensure-Venv {
    if (!(Test-Path ".\.venv")) {
        Write-Host "[dev] Creating Python venv..." -ForegroundColor Cyan
        python -m venv .venv
    }
    Write-Host "[dev] Activating venv..." -ForegroundColor Cyan
    $env:VIRTUAL_ENV = Join-Path (Get-Location) ".venv"
    $venvPython = ".\.venv\\Scripts\\python.exe"
    if (!(Test-Path $venvPython)) {
        throw "Python exe not found in venv: $venvPython"
    }
    Write-Host "[dev] Installing ml_service requirements..." -ForegroundColor Cyan
    & $venvPython -m pip install -U pip
    & $venvPython -m pip install -r "ml_service/requirements.txt"
}

function Run-API {
    $venvPython = ".\.venv\\Scripts\\python.exe"
    Write-Host "[dev] Starting ML API (uvicorn) on http://localhost:8000 ..." -ForegroundColor Green
    & $venvPython -m uvicorn ml_service.src.api.main:app --reload --host 0.0.0.0 --port 8000
}

function Compose-Up {
    Write-Host "[dev] docker compose up -d" -ForegroundColor Green
    docker compose up -d
}

function Compose-Down {
    Write-Host "[dev] docker compose down" -ForegroundColor Yellow
    docker compose down
}

function Ingestion-ManualFetch {
    Write-Host "[dev] Running ingestion manualfetch (Go)..." -ForegroundColor Green
    go run "./ingestion/cmd/manualfetch"
}

function Ingestion-Worker {
    Write-Host "[dev] Running ingestion worker (Go)..." -ForegroundColor Green
    go run "./ingestion/cmd/worker"
}

try {
    if ($Compose) { Compose-Up }
    if ($ComposeDown) { Compose-Down }

    if ($API) {
        Ensure-Venv
        Run-API
    }

    if ($ManualFetch) { Ingestion-ManualFetch }
    if ($Worker) { Ingestion-Worker }

    if (-not ($Compose -or $ComposeDown -or $API -or $ManualFetch -or $Worker)) {
        Write-Host "[dev] No switches provided. Defaulting to: Setup venv + run API." -ForegroundColor Cyan
        Ensure-Venv
        Run-API
    }
}
finally {
    Pop-Location
}
