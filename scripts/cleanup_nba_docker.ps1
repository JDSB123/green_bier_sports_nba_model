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
    try {
        $containers = @(docker.exe ps -a --filter "name=nba" --format "{{.Names}}")
        if ($containers -and $containers.Count -gt 0 -and $containers[0] -ne "") {
            foreach ($container in $containers) {
                if ([string]::IsNullOrWhiteSpace($container)) { continue }
                Write-Host "  Removing container: $container" -ForegroundColor Gray
                docker.exe rm -f $container 2>$null | Out-Null
                $removedCount++
            }
            Write-Host "  ✓ Removed containers" -ForegroundColor Green
        } else {
            Write-Host "  No NBA containers found" -ForegroundColor Gray
        }
    } catch {
        Write-Warning "Error listing containers: $_"
    }
}

# Clean up images
if ($All -or $Images) {
    Write-Host "`nCleaning up images..." -ForegroundColor Yellow
    try {
        $images = @(docker.exe images --filter "reference=*nba*" --format "{{.Repository}}:{{.Tag}}")
        if ($images -and $images.Count -gt 0 -and $images[0] -ne "") {
            foreach ($image in $images) {
                if ([string]::IsNullOrWhiteSpace($image)) { continue }
                Write-Host "  Removing image: $image" -ForegroundColor Gray
                docker.exe rmi -f $image 2>$null | Out-Null
                $removedCount++
            }
            Write-Host "  ✓ Removed images" -ForegroundColor Green
        } else {
            Write-Host "  No NBA images found" -ForegroundColor Gray
        }
    } catch {
        Write-Warning "Error listing images: $_"
    }
}

# Clean up volumes
if ($All -or $Volumes) {
    Write-Host "`nCleaning up volumes..." -ForegroundColor Yellow
    try {
        $volumes = @(docker.exe volume ls --filter "name=nba" --format "{{.Name}}")
        if ($volumes -and $volumes.Count -gt 0 -and $volumes[0] -ne "") {
            foreach ($volume in $volumes) {
                if ([string]::IsNullOrWhiteSpace($volume)) { continue }
                Write-Host "  Removing volume: $volume" -ForegroundColor Gray
                docker.exe volume rm $volume 2>$null | Out-Null
                $removedCount++
            }
            Write-Host "  ✓ Removed volumes" -ForegroundColor Green
        } else {
            Write-Host "  No NBA volumes found" -ForegroundColor Gray
        }
    } catch {
        Write-Warning "Error listing volumes: $_"
    }
}

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Green
