#!/usr/bin/env pwsh
# NBA Model Docker Image Cleanup Script
# Removes all images except nbagbsacr.azurecr.io/nba-gbsv-api:v6.10

Write-Host "=== NBA Docker Image Cleanup ===" -ForegroundColor Cyan

# The image we want to KEEP
$keepImage = "nbagbsacr.azurecr.io/nba-gbsv-api:v6.10"

Write-Host "`nCurrent NBA-related images:" -ForegroundColor Yellow
docker images | Select-String "nba"

Write-Host "`n⚠️  This will remove ALL NBA images except: $keepImage" -ForegroundColor Yellow
$confirm = Read-Host "Continue? (y/n)"

if ($confirm -ne 'y') {
    Write-Host "Cleanup cancelled." -ForegroundColor Red
    exit 0
}

Write-Host "`n🗑️  Removing old images..." -ForegroundColor Cyan

# Get all NBA images
$images = docker images --format "{{.Repository}}:{{.Tag}}" | Select-String "nba"

$removed = 0
$kept = 0

foreach ($image in $images) {
    $imageStr = $image.ToString().Trim()
    
    if ($imageStr -eq $keepImage) {
        Write-Host "✅ Keeping: $imageStr" -ForegroundColor Green
        $kept++
    } else {
        Write-Host "🗑️  Removing: $imageStr" -ForegroundColor Red
        docker rmi $imageStr -f 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $removed++
        }
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "✅ Kept: $kept image(s)" -ForegroundColor Green
Write-Host "🗑️  Removed: $removed image(s)" -ForegroundColor Red

Write-Host "`nRemaining NBA images:" -ForegroundColor Yellow
docker images | Select-String "nba"

Write-Host "`n✅ Cleanup complete!" -ForegroundColor Green
