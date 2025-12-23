#!/usr/bin/env pwsh
# Complete NBA Project Cleanup Script
# Cleans GitHub, Azure, and Local environments

Write-Host "=== NBA Project Complete Cleanup ===" -ForegroundColor Cyan

# ===== LOCAL GIT CLEANUP =====
Write-Host "`n📦 Git Repository Cleanup" -ForegroundColor Yellow
Write-Host "Current branch: $(git branch --show-current)"

$response = Read-Host "Clean up old local branches? (y/n)"
if ($response -eq 'y') {
    $oldBranches = @(
        '1h/moneyline', '1h/spread', '1h/totals',
        'fg/moneyline', 'fg/spread', 'fg/totals',
        'q1/moneyline', 'q1/spread', 'q1/totals',
        'data_ingestion/api_basketball', 'data_ingestion/betting_splits',
        'data_ingestion/espn', 'data_ingestion/injuries', 'data_ingestion/the_odds_api',
        'data_ingestion_main', 'segment_pick_main', 'testing'
    )
    
    foreach ($branch in $oldBranches) {
        git branch -d $branch 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Deleted: $branch" -ForegroundColor Green
        }
    }
}

# Handle uncommitted files
Write-Host "`n📝 Uncommitted Files:" -ForegroundColor Yellow
git status --short

$response = Read-Host "`nCommit cleanup script? (y/n)"
if ($response -eq 'y') {
    git add scripts/cleanup_docker_images.ps1
    git add scripts/full_cleanup.ps1
    git commit -m "Add cleanup scripts"
}

if (Test-Path "temp_debug.txt") {
    Remove-Item "temp_debug.txt"
    Write-Host "✅ Removed temp_debug.txt" -ForegroundColor Green
}

# ===== DOCKER CLEANUP =====
Write-Host "`n🐳 Docker Cleanup" -ForegroundColor Yellow
Write-Host "Current Docker usage:"
docker system df

$response = Read-Host "`nRun full Docker cleanup? This will free ~70GB (y/n)"
if ($response -eq 'y') {
    Write-Host "Removing stopped containers..." -ForegroundColor Cyan
    docker container prune -f
    
    Write-Host "Removing unused images..." -ForegroundColor Cyan
    docker image prune -a -f
    
    Write-Host "Removing unused volumes..." -ForegroundColor Cyan
    docker volume prune -f
    
    Write-Host "Removing build cache..." -ForegroundColor Cyan
    docker builder prune -a -f
    
    Write-Host "`n✅ Docker cleanup complete!" -ForegroundColor Green
    docker system df
}

# ===== AZURE CLEANUP =====
Write-Host "`n☁️  Azure Cleanup" -ForegroundColor Yellow
Write-Host "Checking Azure resources..."

# List duplicate resources
Write-Host "`n⚠️  Duplicate NBA resources found in greenbier-enterprise-rg:" -ForegroundColor Red
az resource list --resource-group greenbier-enterprise-rg --query "[?contains(name, 'nba')].{Name:name, Type:type}" -o table

$response = Read-Host "`nDelete duplicate NBA resources from greenbier-enterprise-rg? (y/n)"
if ($response -eq 'y') {
    Write-Host "Deleting duplicate nba-gbsv-api..." -ForegroundColor Cyan
    az containerapp delete --name nba-gbsv-api --resource-group greenbier-enterprise-rg --yes 2>&1 | Out-Null
    
    Write-Host "Deleting greenbier-nba-env..." -ForegroundColor Cyan
    az containerapp env delete --name greenbier-nba-env --resource-group greenbier-enterprise-rg --yes 2>&1 | Out-Null
    
    Write-Host "✅ Azure duplicates cleaned" -ForegroundColor Green
}

# Check wrong registry
$response = Read-Host "`nDelete old greenbieracr registry? (y/n)"
if ($response -eq 'y') {
    az acr delete --name greenbieracr --resource-group greenbier-enterprise-rg --yes
    Write-Host "✅ Deleted greenbieracr" -ForegroundColor Green
}

# ===== SUMMARY =====
Write-Host "`n=== Cleanup Summary ===" -ForegroundColor Cyan
Write-Host "✅ Active Resource Group: nba-gbsv-model-rg" -ForegroundColor Green
Write-Host "✅ Active Registry: nbagbsacr.azurecr.io" -ForegroundColor Green
Write-Host "✅ Active Image: nba-gbsv-api:v6.10" -ForegroundColor Green
Write-Host "✅ Active Container App: nba-gbsv-api (nba-gbsv-model-rg)" -ForegroundColor Green

Write-Host "`n🎉 Cleanup complete!" -ForegroundColor Green
