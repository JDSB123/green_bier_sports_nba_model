#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deep Git cleanup - Remove large files from history

.DESCRIPTION
    ⚠️ WARNING: This script REWRITES Git history!
    
    This script removes large files that were accidentally committed:
    - data/external/kaggle/nba_2008-2025.csv (2.4 MB)
    - coverage.xml (247 KB, multiple versions)
    
    Impact:
    - Repo size reduced by ~40%
    - Faster clones
    - Cleaner history
    
    ⚠️ ALL TEAM MEMBERS MUST RE-CLONE AFTER THIS!

.PARAMETER Backup
    Create backup before cleanup (default: true)

.PARAMETER DryRun
    Show what would be removed without making changes

.EXAMPLE
    .\git-deep-cleanup.ps1
    Full cleanup with backup

.EXAMPLE
    .\git-deep-cleanup.ps1 -DryRun
    Preview what would be removed

.EXAMPLE
    .\git-deep-cleanup.ps1 -Backup:$false
    Skip backup (not recommended)
#>

param(
    [switch]$DryRun,
    [bool]$Backup = $true
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Danger {
    param([string]$Message)
    Write-Host "❌ $MESSAGE" -ForegroundColor Red
}

Write-Step "Git Deep Cleanup - Remove Large Files"

Write-Host "⚠️  WARNING: This will REWRITE Git history!" -ForegroundColor Red
Write-Host "⚠️  All team members must re-clone after this!" -ForegroundColor Red
Write-Host ""

# Check if git-filter-repo is installed
Write-Step "[1/7] Checking prerequisites"

$filterRepoInstalled = $false
try {
    git filter-repo --version 2>&1 | Out-Null
    $filterRepoInstalled = $true
    Write-Success "git-filter-repo is installed"
} catch {
    Write-Warning-Custom "git-filter-repo not found"
}

if (-not $filterRepoInstalled) {
    Write-Host ""
    Write-Host "Installing git-filter-repo..." -ForegroundColor Yellow
    
    if ($DryRun) {
        Write-Host "  [DRY RUN] Would install: pip install git-filter-repo" -ForegroundColor Gray
    } else {
        try {
            pip install git-filter-repo
            Write-Success "git-filter-repo installed"
        } catch {
            Write-Danger "Failed to install git-filter-repo"
            Write-Host "Please install manually: pip install git-filter-repo"
            exit 1
        }
    }
}

# Check repo state
Write-Step "[2/7] Checking repository state"

$status = git status --porcelain
if ($status) {
    Write-Danger "Working directory is not clean!"
    git status --short
    Write-Host ""
    Write-Host "Please commit or stash changes before running deep cleanup."
    exit 1
}
Write-Success "Working directory clean"

# Create backup
if ($Backup) {
    Write-Step "[3/7] Creating backup"
    
    $backupPath = "..\NBA_main_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    if ($DryRun) {
        Write-Host "  [DRY RUN] Would create backup at: $backupPath" -ForegroundColor Gray
    } else {
        Write-Host "  Creating backup at: $backupPath" -ForegroundColor Yellow
        git clone --mirror . $backupPath
        Write-Success "Backup created at: $backupPath"
    }
} else {
    Write-Step "[3/7] Skipping backup (not recommended!)"
    Write-Warning-Custom "No backup will be created"
}

# Analyze large files
Write-Step "[4/7] Analyzing large files in history"

Write-Host "  Finding largest files..." -ForegroundColor Yellow
$largeFiles = git rev-list --all --objects | 
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | 
    Where-Object { $_ -match '^blob' } | 
    Sort-Object { [int]($_ -split ' ')[2] } -Descending | 
    Select-Object -First 20

Write-Host "`n  Top 20 largest files in history:" -ForegroundColor White
$largeFiles | ForEach-Object {
    $parts = $_ -split ' ', 4
    $size = [int]$parts[2]
    $path = $parts[3]
    $sizeMB = [math]::Round($size / 1MB, 2)
    Write-Host "    $sizeMB MB - $path" -ForegroundColor Gray
}

# Files to remove
Write-Step "[5/7] Files to remove"

$filesToRemove = @(
    "data/external/kaggle/nba_2008-2025.csv",
    "coverage.xml"
)

Write-Host "  These files will be PERMANENTLY removed from history:" -ForegroundColor Red
foreach ($file in $filesToRemove) {
    Write-Host "    ❌ $file" -ForegroundColor Red
}

# Confirm
Write-Step "[6/7] Confirmation"

if ($DryRun) {
    Write-Host "[DRY RUN] No changes will be made" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "⚠️  This action CANNOT be undone!" -ForegroundColor Red
    Write-Host "⚠️  All team members will need to re-clone!" -ForegroundColor Red
    Write-Host ""
    $confirm = Read-Host "Type 'YES' to proceed with deep cleanup"
    
    if ($confirm -ne "YES") {
        Write-Warning-Custom "Deep cleanup cancelled"
        exit 0
    }
}

# Execute cleanup
Write-Step "[7/7] Executing cleanup"

if ($DryRun) {
    Write-Host "[DRY RUN] Would execute the following:" -ForegroundColor Yellow
    Write-Host ""
    foreach ($file in $filesToRemove) {
        Write-Host "  git filter-repo --path $file --invert-paths --force" -ForegroundColor Gray
    }
    Write-Host "  git reflog expire --expire=now --all" -ForegroundColor Gray
    Write-Host "  git gc --prune=now --aggressive" -ForegroundColor Gray
    Write-Host "  git push origin --force --all" -ForegroundColor Gray
    Write-Host "  git push origin --force --tags" -ForegroundColor Gray
} else {
    # Remove files
    Write-Host "  Removing files from history..." -ForegroundColor Yellow
    
    foreach ($file in $filesToRemove) {
        Write-Host "    Processing: $file" -ForegroundColor Gray
        git filter-repo --path $file --invert-paths --force
    }
    
    Write-Success "Files removed from history"
    
    # Cleanup
    Write-Host "`n  Cleaning up repository..." -ForegroundColor Yellow
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    Write-Success "Repository cleaned"
    
    # Get size before/after
    $sizeAfter = (Get-ChildItem .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "`n  New repository size: $([math]::Round($sizeAfter, 2)) MB" -ForegroundColor Green
    
    # Push
    Write-Host "`n  Pushing cleaned history..." -ForegroundColor Yellow
    Write-Warning-Custom "This will force-push to origin!"
    
    $confirmPush = Read-Host "Push to remote? (y/N)"
    if ($confirmPush -eq "y") {
        git push origin --force --all
        git push origin --force --tags
        Write-Success "Pushed cleaned history to remote"
    } else {
        Write-Warning-Custom "Skipped push - run manually: git push origin --force --all"
    }
}

# Summary
Write-Step "Deep Cleanup Complete!"

if ($DryRun) {
    Write-Host "[DRY RUN] No changes were made" -ForegroundColor Yellow
    Write-Host "Run without -DryRun to execute cleanup" -ForegroundColor Yellow
} else {
    Write-Success "Repository cleaned successfully!"
    
    if ($Backup) {
        Write-Host "`n📦 Backup location: $backupPath" -ForegroundColor Cyan
    }
    
    Write-Host "`n⚠️  IMPORTANT: Team members must re-clone!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Send this message to your team:" -ForegroundColor Yellow
    Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "The NBA model repository has been cleaned up." -ForegroundColor White
    Write-Host "You must re-clone to get the cleaned history:" -ForegroundColor White
    Write-Host ""
    Write-Host "1. Backup your local changes (if any)" -ForegroundColor White
    Write-Host "2. Delete your local clone" -ForegroundColor White
    Write-Host "3. Fresh clone: git clone https://github.com/JDSB123/green_bier_sports_nba_model.git" -ForegroundColor White
    Write-Host "4. Benefits: ~40% smaller repo, faster operations" -ForegroundColor White
    Write-Host "─────────────────────────────────────────────────────────" -ForegroundColor Gray
}

Write-Host ""
