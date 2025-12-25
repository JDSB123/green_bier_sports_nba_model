# NBA Production Deployment Script
# Automates the flow: Git -> Docker -> Azure
# Usage: ./scripts/deploy_production.ps1 [-Version "NBA_v33.0.X.X"]

param(
    [string]$Version
)

$ErrorActionPreference = "Stop"

# 1. Determine Version
if (-not $Version) {
    if (Test-Path "VERSION") {
        $Version = (Get-Content "VERSION").Trim()
    } else {
        Write-Error "No version specified and no VERSION file found."
        exit 1
    }
}

Write-Host "🚀 STARTING DEPLOYMENT FOR VERSION: $Version" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 2. Update VERSION file if provided argument differs
if ((Test-Path "VERSION") -and ((Get-Content "VERSION").Trim() -ne $Version)) {
    Set-Content -Path "VERSION" -Value $Version
    Write-Host "📝 Updated VERSION file to $Version" -ForegroundColor Gray
}

# 3. Git Sync
Write-Host "`n📦 STEP 1: Git Sync" -ForegroundColor Yellow
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "  Committing changes..." -ForegroundColor Gray
    git add .
    git commit -m "Release $Version"
}
Write-Host "  Pushing to GitHub..." -ForegroundColor Gray
git push origin main

# 4. Docker Build
$ImageTag = "nbagbsacr.azurecr.io/nba-gbsv-api:$Version"
$LatestTag = "nbagbsacr.azurecr.io/nba-gbsv-api:latest"

Write-Host "`n🐳 STEP 2: Docker Build" -ForegroundColor Yellow
Write-Host "  Building $ImageTag..." -ForegroundColor Gray
docker build -t $ImageTag -t $LatestTag -f Dockerfile.combined .

# 5. Docker Push
Write-Host "`n☁️ STEP 3: Push to Registry" -ForegroundColor Yellow
Write-Host "  Logging into ACR..." -ForegroundColor Gray
az acr login -n nbagbsacr
Write-Host "  Pushing images..." -ForegroundColor Gray
docker push $ImageTag
docker push $LatestTag

# 6. Azure Deployment
Write-Host "`n🚀 STEP 4: Azure Deployment" -ForegroundColor Yellow
Write-Host "  Updating Container App to $ImageTag..." -ForegroundColor Gray
az containerapp update -n nba-gbsv-api -g nba-gbsv-model-rg --image $ImageTag --set-env-vars NBA_MODEL_VERSION=$Version

# 7. Verification
Write-Host "`n✅ STEP 5: Verification" -ForegroundColor Yellow
$healthUrl = "https://nba-gbsv-api.livelycoast-b48c3cb0.eastus.azurecontainerapps.io/health"
Write-Host "  Checking health at $healthUrl..." -ForegroundColor Gray
try {
    $response = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 30
    Write-Host "  Status: Healthy" -ForegroundColor Green
    Write-Host "  Version: $($response.version)" -ForegroundColor Green
    
    if ($response.version -ne $Version) {
        Write-Warning "  ⚠️ Mismatch! Deployed version ($($response.version)) does not match target ($Version)."
    }
} catch {
    Write-Error "  ❌ Health check failed: $_"
}

Write-Host "`n✨ Deployment Complete!" -ForegroundColor Cyan
