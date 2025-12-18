# NBA Prediction System v5.0 - Single Container Production Entry Point
# Builds and runs the hardened production Docker image with baked-in models.
# Produces "Model vs Market" comparisons and saves a report to data/processed/.

param(
    [string]$Date = "today",
    [string]$Matchup = ""
)

$ErrorActionPreference = "Stop"

Write-Host "NBA Prediction System v5.0 - Single Container Production" -ForegroundColor Cyan
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

# Image name and tag
$IMAGE_NAME = "nba-strict-api"
$IMAGE_TAG = "latest"
$CONTAINER_NAME = "nba-api"
$FULL_IMAGE = "${IMAGE_NAME}:${IMAGE_TAG}"

# Check if .env file exists
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

Write-Host "Building production Docker image..." -ForegroundColor Yellow
docker build -t $FULL_IMAGE -f Dockerfile .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build Docker image"
    exit $LASTEXITCODE
}
Write-Host "✓ Image built successfully" -ForegroundColor Green
Write-Host ""

# Stop and remove existing container if it exists
Write-Host "Stopping existing container (if any)..." -ForegroundColor Yellow
docker stop $CONTAINER_NAME 2>$null | Out-Null
docker rm $CONTAINER_NAME 2>$null | Out-Null

# Start the single container
Write-Host "Starting production container..." -ForegroundColor Yellow
$dockerRunArgs = @(
    "run",
    "-d",
    "--name", $CONTAINER_NAME,
    "-p", "8090:8080",
    "--env-file", $envFile,
    "--restart", "unless-stopped"
)

# Mount secrets directory if it exists (read-only)
if (Test-Path "secrets") {
    $dockerRunArgs += "--mount"
    $dockerRunArgs += "type=bind,source=$(Resolve-Path secrets),target=/run/secrets,readonly"
    Write-Host "  Mounting secrets directory (read-only)" -ForegroundColor Gray
}

# Add the image name
$dockerRunArgs += $FULL_IMAGE

& docker $dockerRunArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start Docker container"
    exit $LASTEXITCODE
}

Write-Host "✓ Container started" -ForegroundColor Green
Write-Host ""

# Wait for container to be healthy
Write-Host "Waiting for API to be ready..." -ForegroundColor Yellow
$maxAttempts = 30
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    Start-Sleep -Seconds 2
    $attempt++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8090/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $healthData = $response.Content | ConvertFrom-Json
            if ($healthData.engine_loaded -eq $true) {
                $ready = $true
                Write-Host "✓ API is healthy and engine is loaded" -ForegroundColor Green
            } else {
                Write-Host "  Attempt ${attempt}/${maxAttempts}: Engine not loaded yet..." -ForegroundColor Gray
            }
        }
    } catch {
        Write-Host "  Attempt ${attempt}/${maxAttempts}: Waiting for API..." -ForegroundColor Gray
    }
}

if (-not $ready) {
    Write-Error "API did not become ready within timeout. Check logs: docker logs $CONTAINER_NAME"
    exit 1
}

Write-Host ""

# Build arguments for analyze_slate_docker.py
$analysisArgs = @()
if ($Date) {
    $analysisArgs += "--date"
    $analysisArgs += $Date
}
if ($Matchup) {
    $analysisArgs += "--matchup"
    $analysisArgs += $Matchup
}

# Run the analysis script
Write-Host "Running analysis..." -ForegroundColor Cyan
Write-Host ""
python scripts/analyze_slate_docker.py $analysisArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Analysis failed. Check logs: docker logs $CONTAINER_NAME" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "✓ Analysis complete. Reports saved to data/processed/" -ForegroundColor Green
