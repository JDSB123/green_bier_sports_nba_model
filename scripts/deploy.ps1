#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy NBA prediction API to Azure Container App with version from VERSION file

.DESCRIPTION
    This script automates the NBA model deployment pipeline:
    1. Reads version from VERSION file (single source of truth)
    2. Ensures git is clean and pushed to GitHub
    3. Builds Docker image with correct version tag
    4. Pushes to Azure Container Registry
    5. Updates Azure Container App
    6. Verifies deployment health

.PARAMETER SkipGitCheck
    Skip git status verification (not recommended for production)

.PARAMETER SkipBuild
    Skip Docker build (use existing image)

.PARAMETER DryRun
    Show commands without executing

.EXAMPLE
    .\deploy.ps1
    Standard deployment flow

.EXAMPLE
    .\deploy.ps1 -DryRun
    Preview deployment commands

.NOTES
    Author: NBA Prediction System
    Version: 1.0.0
    Requires: Docker, Azure CLI, git
#>

param(
    [switch]$SkipGitCheck,
    [switch]$SkipBuild,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Constants
$RESOURCE_GROUP = "nba-gbsv-model-rg"
$CONTAINER_APP = "nba-gbsv-api"
$ACR_NAME = "nbagbsacr"
$DOCKERFILE = "Dockerfile.combined"
$HEALTH_URL = "https://nba-gbsv-api.ambitiouscoast-4bcd4cd8.eastus.azurecontainerapps.io/health"

# Helper functions
function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Execute-Command {
    param([string]$Command)
    
    Write-Host "  → $Command" -ForegroundColor Gray
    
    if (-not $DryRun) {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "Command failed with exit code $LASTEXITCODE"
            exit $LASTEXITCODE
        }
    }
}

# Main deployment logic
try {
    Write-Step "NBA Model Deployment - Automated Pipeline"

    # Step 1: Read VERSION file
    Write-Step "[1/7] Reading VERSION file"
    
    $VERSION_FILE = Join-Path $PSScriptRoot ".." "VERSION"
    if (-not (Test-Path $VERSION_FILE)) {
        Write-Failure "VERSION file not found at $VERSION_FILE"
        exit 1
    }
    
    $VERSION = (Get-Content $VERSION_FILE -Raw).Trim()
    Write-Success "Version: $VERSION"
    
    $IMAGE_NAME = "$ACR_NAME.azurecr.io/nba-gbsv-api:$VERSION"
    Write-Host "  Image: $IMAGE_NAME" -ForegroundColor Gray

    # Step 2: Git status check
    if (-not $SkipGitCheck) {
        Write-Step "[2/7] Checking git status"
        
        $GitStatus = git status --porcelain
        if ($GitStatus) {
            Write-Warning-Custom "Uncommitted changes detected:"
            git status --short
            Write-Host ""
            $Response = Read-Host "Continue anyway? (y/N)"
            if ($Response -ne "y") {
                Write-Failure "Deployment cancelled - commit changes first"
                exit 1
            }
        } else {
            Write-Success "Working directory clean"
        }
        
        # Check if pushed to GitHub
        $LocalCommit = git rev-parse HEAD
        $RemoteCommit = git rev-parse origin/main
        
        if ($LocalCommit -ne $RemoteCommit) {
            Write-Warning-Custom "Local branch is ahead/behind origin/main"
            git status -sb
            Write-Host ""
            $Response = Read-Host "Push to GitHub before deploying? (Y/n)"
            if ($Response -ne "n") {
                Execute-Command "git push origin main"
                Write-Success "Pushed to GitHub"
            }
        } else {
            Write-Success "In sync with origin/main"
        }
    } else {
        Write-Warning-Custom "Skipping git checks (--SkipGitCheck)"
    }

    # Step 3: Docker build
    if (-not $SkipBuild) {
        Write-Step "[3/7] Building Docker image"
        
        Execute-Command "docker build -t $IMAGE_NAME -f $DOCKERFILE ."
        Write-Success "Docker image built: $IMAGE_NAME"
    } else {
        Write-Warning-Custom "Skipping Docker build (--SkipBuild)"
    }

    # Step 4: Azure CLI login check
    Write-Step "[4/7] Checking Azure CLI authentication"
    
    $AzAccount = az account show 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning-Custom "Not logged into Azure CLI"
        Execute-Command "az login"
    } else {
        Write-Success "Azure CLI authenticated"
    }

    # Step 5: Push to ACR
    Write-Step "[5/7] Pushing to Azure Container Registry"
    
    Execute-Command "az acr login -n $ACR_NAME"
    Execute-Command "docker push $IMAGE_NAME"
    Write-Success "Image pushed to ACR: $IMAGE_NAME"

    # Step 6: Update Container App
    Write-Step "[6/7] Updating Azure Container App"
    
    Execute-Command "az containerapp update -n $CONTAINER_APP -g $RESOURCE_GROUP --image $IMAGE_NAME"
    Write-Success "Container App updated"

    # Step 7: Health check
    Write-Step "[7/7] Verifying deployment"
    
    Write-Host "  Waiting 10 seconds for container to start..." -ForegroundColor Gray
    if (-not $DryRun) {
        Start-Sleep -Seconds 10
    }
    
    try {
        $Response = Invoke-WebRequest -Uri $HEALTH_URL -UseBasicParsing
        $HealthData = $Response.Content | ConvertFrom-Json
        
        if ($HealthData.status -eq "healthy" -and $HealthData.version -eq $VERSION) {
            Write-Success "Deployment verified successfully"
            Write-Host ""
            Write-Host "  Status:  $($HealthData.status)" -ForegroundColor Green
            Write-Host "  Version: $($HealthData.version)" -ForegroundColor Green
            Write-Host "  URL:     $HEALTH_URL" -ForegroundColor Cyan
        } else {
            Write-Warning-Custom "Health check returned unexpected response"
            Write-Host "  Response: $($Response.Content)"
        }
    } catch {
        Write-Failure "Health check failed: $_"
        Write-Host "  URL: $HEALTH_URL"
        Write-Host "  Check logs: az containerapp logs show -n $CONTAINER_APP -g $RESOURCE_GROUP"
        exit 1
    }

    # Success summary
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "   DEPLOYMENT SUCCESSFUL" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Version:  $VERSION" -ForegroundColor White
    Write-Host "  Image:    $IMAGE_NAME" -ForegroundColor White
    Write-Host "  Endpoint: $HEALTH_URL" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Test predictions: python scripts/run_slate.py" -ForegroundColor Gray
    Write-Host "  2. View logs:        az containerapp logs show -n $CONTAINER_APP -g $RESOURCE_GROUP" -ForegroundColor Gray
    Write-Host "  3. Monitor metrics:  az monitor metrics list --resource ..." -ForegroundColor Gray
    Write-Host ""

} catch {
    Write-Failure "Deployment failed: $_"
    exit 1
}
