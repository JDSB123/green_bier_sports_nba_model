# Deploy Functions with Dependencies
# This script copies necessary source code and deploys

param(
    [string]$FunctionAppName = "green-bier-sports-nba",
    [string]$ResourceGroup = "green-bier-sport-ventures-rg"
)

$ErrorActionPreference = "Continue"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$functionAppDir = Join-Path $scriptRoot "function_app"

Write-Host "[DEPLOY] Deploying Function App with dependencies..." -ForegroundColor Cyan
Write-Host ""

# Step 1: Copy source code to function_app directory
Write-Host "[STEP 1] Copying source code..." -ForegroundColor Yellow

# Create directories in function_app
New-Item -ItemType Directory -Force -Path (Join-Path $functionAppDir "src") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $functionAppDir "scripts") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $functionAppDir "data\processed\models") | Out-Null

# Copy src directory
Write-Host "  Copying src/..." -ForegroundColor Cyan
Copy-Item -Path (Join-Path $projectRoot "src\*") -Destination (Join-Path $functionAppDir "src") -Recurse -Force -ErrorAction SilentlyContinue

# Copy scripts directory (but filter large files)
Write-Host "  Copying scripts/..." -ForegroundColor Cyan
Get-ChildItem -Path (Join-Path $projectRoot "scripts") -File -Filter "*.py" | Copy-Item -Destination (Join-Path $functionAppDir "scripts") -Force -ErrorAction SilentlyContinue

# Copy models if they exist
Write-Host "  Copying models/..." -ForegroundColor Cyan
$modelsSource = Join-Path $projectRoot "data\processed\models"
if (Test-Path $modelsSource) {
    Copy-Item -Path "$modelsSource\*" -Destination (Join-Path $functionAppDir "data\processed\models") -Recurse -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "  [WARN] Models directory not found at: $modelsSource" -ForegroundColor Yellow
}

Write-Host "[OK] Source code copied" -ForegroundColor Green
Write-Host ""

# Step 2: Update .funcignore to NOT ignore src and scripts
Write-Host "[STEP 2] Updating .funcignore..." -ForegroundColor Yellow
$funcignorePath = Join-Path $functionAppDir ".funcignore"
$funcignoreContent = @"
.venv
__pycache__
*.pyc
.pytest_cache
.coverage
"@
$funcignoreContent | Out-File -FilePath $funcignorePath -Encoding UTF8
Write-Host "[OK] .funcignore updated" -ForegroundColor Green
Write-Host ""

# Step 3: Deploy
Write-Host "[STEP 3] Deploying to Azure..." -ForegroundColor Yellow
Write-Host "  This will take 3-5 minutes..." -ForegroundColor Gray
Push-Location $functionAppDir
func azure functionapp publish $FunctionAppName --python --build remote
$deployResult = $LASTEXITCODE
Pop-Location

if ($deployResult -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] Deployment complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your Function App URL: https://${FunctionAppName}.azurewebsites.net" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Test it:" -ForegroundColor Yellow
    Write-Host "  .\azure\test_picks.ps1" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "[ERROR] Deployment failed!" -ForegroundColor Red
    Write-Host "Check the output above for errors." -ForegroundColor Yellow
}
