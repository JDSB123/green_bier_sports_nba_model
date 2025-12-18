# NBA Docker Cleanup Script
# Safely removes old NBA containers, images, and volumes
# Only removes resources with 'nba' in the name to avoid affecting other Docker resources

param(
    [switch]$Force,
    [switch]$Images,
    [switch]$Containers,
    [switch]$Volumes,
    [switch]$All
)

$ErrorActionPreference = "Stop"

Write-Host "NBA Docker Cleanup Script" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

if (-not $Force -and -not $All) {
    Write-Host "This script will remove NBA-related Docker resources." -ForegroundColor Yellow
    Write-Host "Use -Force to skip confirmation, or specify what to clean:" -ForegroundColor Yellow
    Write-Host "  -Containers  : Remove stopped containers" -ForegroundColor Gray
    Write-Host "  -Images      : Remove images" -ForegroundColor Gray
    Write-Host "  -Volumes     : Remove volumes" -ForegroundColor Gray
    Write-Host "  -All         : Remove all of the above" -ForegroundColor Gray
    Write-Host ""
    $confirm = Read-Host "Continue? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
}

$removedCount = 0

# Clean up containers
if ($All -or $Containers) {
    Write-Host "`nCleaning up containers..." -ForegroundColor Yellow
    $containers = docker ps -a --filter "name=nba" --format "{{.Names}}"
    if ($containers) {
        foreach ($container in $containers) {
            Write-Host "  Removing container: $container" -ForegroundColor Gray
            docker rm -f $container 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $removedCount++
            }
        }
        Write-Host "  ✓ Removed $removedCount container(s)" -ForegroundColor Green
    } else {
        Write-Host "  No NBA containers found" -ForegroundColor Gray
    }
}

# Clean up images
if ($All -or $Images) {
    Write-Host "`nCleaning up images..." -ForegroundColor Yellow
    $images = docker images --filter "reference=*nba*" --format "{{.Repository}}:{{.Tag}}"
    if ($images) {
        foreach ($image in $images) {
            Write-Host "  Removing image: $image" -ForegroundColor Gray
            docker rmi -f $image 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $removedCount++
            }
        }
        Write-Host "  ✓ Removed $removedCount image(s)" -ForegroundColor Green
    } else {
        Write-Host "  No NBA images found" -ForegroundColor Gray
    }
}

# Clean up volumes
if ($All -or $Volumes) {
    Write-Host "`nCleaning up volumes..." -ForegroundColor Yellow
    $volumes = docker volume ls --filter "name=nba" --format "{{.Name}}"
    if ($volumes) {
        foreach ($volume in $volumes) {
            Write-Host "  Removing volume: $volume" -ForegroundColor Gray
            docker volume rm $volume 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                $removedCount++
            }
        }
        Write-Host "  ✓ Removed $removedCount volume(s)" -ForegroundColor Green
    } else {
        Write-Host "  No NBA volumes found" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To rebuild and start fresh:" -ForegroundColor Cyan
Write-Host "  ./run.ps1" -ForegroundColor White
