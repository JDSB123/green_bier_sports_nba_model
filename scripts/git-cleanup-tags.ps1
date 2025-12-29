#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick Git cleanup - delete old tags and create proper release tag

.DESCRIPTION
    This script performs LOW-RISK cleanup:
    - Deletes old v6.x tags (outdated versioning scheme)
    - Creates NBA_v33.0.8.0 tag (current version)
    - Does NOT rewrite history
    - Does NOT delete branches
    - Safe to run, can be undone

.PARAMETER DryRun
    Show what would be done without making changes

.EXAMPLE
    .\git-cleanup-tags.ps1
    Quick cleanup with proper tagging

.EXAMPLE
    .\git-cleanup-tags.ps1 -DryRun
    Preview changes without executing
#>

param(
    [switch]$DryRun
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

Write-Step "Git Tag Cleanup - Quick & Safe"

# Get current version
$VERSION = (Get-Content VERSION -Raw).Trim()
Write-Host "Current version: $VERSION" -ForegroundColor White

# List old tags
Write-Step "[1/3] Finding old tags to delete"

$OLD_TAGS = @("v6.0.0-hardened", "v6.0.1", "v6.2", "v6.4")
$EXISTING_OLD_TAGS = @()

foreach ($tag in $OLD_TAGS) {
    if (git tag -l $tag) {
        $EXISTING_OLD_TAGS += $tag
        Write-Host "  Found: $tag" -ForegroundColor Yellow
    }
}

if ($EXISTING_OLD_TAGS.Count -eq 0) {
    Write-Success "No old tags found - already clean!"
} else {
    Write-Host "`n  Will delete $($EXISTING_OLD_TAGS.Count) old tag(s)" -ForegroundColor Yellow
}

# Check if current version is tagged
Write-Step "[2/3] Checking current version tag"

$CURRENT_TAG_EXISTS = git tag -l $VERSION
if ($CURRENT_TAG_EXISTS) {
    Write-Success "Tag $VERSION already exists"
    $CREATE_TAG = $false
} else {
    Write-Host "  Tag $VERSION does not exist - will create" -ForegroundColor Yellow
    $CREATE_TAG = $true
}

# Confirm
Write-Step "[3/3] Summary"

Write-Host "Actions to perform:"
Write-Host ""
if ($EXISTING_OLD_TAGS.Count -gt 0) {
    Write-Host "DELETE OLD TAGS:" -ForegroundColor Red
    foreach ($tag in $EXISTING_OLD_TAGS) {
        Write-Host "  - $tag" -ForegroundColor Red
    }
    Write-Host ""
}

if ($CREATE_TAG) {
    Write-Host "CREATE NEW TAG:" -ForegroundColor Green
    Write-Host "  - $VERSION" -ForegroundColor Green
    Write-Host ""
}

if (-not $DryRun) {
    Write-Host ""
    $Confirm = Read-Host "Proceed with cleanup? (y/N)"
    if ($Confirm -ne "y") {
        Write-Warning-Custom "Cleanup cancelled"
        exit 0
    }
}

# Execute
Write-Host ""
if ($DryRun) {
    Write-Host "[DRY RUN] Would execute:" -ForegroundColor Yellow
} else {
    Write-Host "Executing cleanup..." -ForegroundColor White
}

# Delete old tags
if ($EXISTING_OLD_TAGS.Count -gt 0) {
    Write-Host "`nDeleting old tags..." -ForegroundColor Yellow
    
    foreach ($tag in $EXISTING_OLD_TAGS) {
        if ($DryRun) {
            Write-Host "  [DRY RUN] git tag -d $tag" -ForegroundColor Gray
            Write-Host "  [DRY RUN] git push origin :refs/tags/$tag" -ForegroundColor Gray
        } else {
            # Delete local
            git tag -d $tag
            Write-Host "  ✅ Deleted local tag: $tag" -ForegroundColor Green
            
            # Delete remote
            try {
                git push origin ":refs/tags/$tag" 2>$null
                Write-Host "  ✅ Deleted remote tag: $tag" -ForegroundColor Green
            } catch {
                Write-Warning-Custom "Could not delete remote tag $tag (may not exist on remote)"
            }
        }
    }
}

# Create new tag
if ($CREATE_TAG) {
    Write-Host "`nCreating new tag: $VERSION" -ForegroundColor Green
    
    $TAG_MESSAGE = @"
NBA Model $VERSION - Production Release

This is the first properly versioned release using the NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD> scheme.

Features:
- 6 markets (1H + FG spreads/totals/moneylines)
- Automated deployment pipeline (scripts/deploy.ps1)
- Version management automation (scripts/bump_version.py)
- CI/CD version validation (.github/workflows/version-check.yml)
- Archive folder for historical tracking
- Complete versioning documentation (VERSIONING.md)

Breaking Changes:
- Removed old v6.x versioning scheme
- All future versions follow VERSIONING.md guidelines

Documentation:
- VERSIONING.md - Version management guide
- CLEANUP_SUMMARY_2025-12-29.md - Repository cleanup summary
- GIT_CLEANUP_PLAN.md - Git cleanup strategy

Deployment:
  .\scripts\deploy.ps1

For version bump:
  python scripts\bump_version.py NBA_vX.X.X.X
"@

    if ($DryRun) {
        Write-Host "  [DRY RUN] git tag -a $VERSION -m '...'" -ForegroundColor Gray
        Write-Host "  [DRY RUN] git push origin $VERSION" -ForegroundColor Gray
    } else {
        git tag -a $VERSION -m $TAG_MESSAGE
        Write-Success "Created local tag: $VERSION"
        
        git push origin $VERSION
        Write-Success "Pushed tag to remote: $VERSION"
    }
}

# Summary
Write-Step "Cleanup Complete!"

if ($DryRun) {
    Write-Host "[DRY RUN] No changes were made" -ForegroundColor Yellow
    Write-Host "Run without -DryRun to apply changes" -ForegroundColor Yellow
} else {
    Write-Success "Git tags cleaned up successfully!"
    
    Write-Host "`nNext steps:"
    Write-Host "  1. Create GitHub Release: https://github.com/JDSB123/green_bier_sports_nba_model/releases/new?tag=$VERSION" -ForegroundColor Cyan
    Write-Host "  2. Add CHANGELOG.md (see GIT_CLEANUP_PLAN.md)" -ForegroundColor Cyan
    Write-Host "  3. Consider deep cleanup to remove large files (see GIT_CLEANUP_PLAN.md)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Current tags:" -ForegroundColor White
git tag -l | Sort-Object
