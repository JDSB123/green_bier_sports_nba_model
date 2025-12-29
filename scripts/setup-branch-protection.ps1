#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Setup GitHub branch protection rules for main branch

.DESCRIPTION
    Configures branch protection on the main branch:
    - Require pull request reviews
    - Require status checks (CI)
    - Prevent force pushes
    - Prevent deletions
    
    Requires GitHub CLI (gh) and admin permissions

.EXAMPLE
    .\setup-branch-protection.ps1
    Setup protection rules with defaults

.EXAMPLE
    .\setup-branch-protection.ps1 -BypassReviews
    Setup without requiring pull request reviews (solo dev mode)
#>

param(
    [switch]$BypassReviews
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   GITHUB BRANCH PROTECTION SETUP" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if gh CLI is installed
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow
try {
    $ghVersion = gh --version
    Write-Host "✅ GitHub CLI installed" -ForegroundColor Green
} catch {
    Write-Host "❌ GitHub CLI not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install GitHub CLI: https://cli.github.com/" -ForegroundColor Yellow
    Write-Host "Or run: winget install GitHub.cli" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
Write-Host "🔍 Checking authentication..." -ForegroundColor Yellow
try {
    $ghAuth = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Not authenticated"
    }
    Write-Host "✅ Authenticated to GitHub" -ForegroundColor Green
} catch {
    Write-Host "❌ Not authenticated to GitHub" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please authenticate: gh auth login" -ForegroundColor Yellow
    exit 1
}

# Get repository info
Write-Host "🔍 Detecting repository..." -ForegroundColor Yellow
$repoInfo = gh repo view --json nameWithOwner,defaultBranchRef | ConvertFrom-Json
$repo = $repoInfo.nameWithOwner
$branch = $repoInfo.defaultBranchRef.name

Write-Host "✅ Repository: $repo" -ForegroundColor Green
Write-Host "✅ Default branch: $branch" -ForegroundColor Green
Write-Host ""

# Confirm
Write-Host "⚠️  This will enable branch protection on: $branch" -ForegroundColor Yellow
Write-Host ""
Write-Host "Protection Rules:" -ForegroundColor Cyan
if (-not $BypassReviews) {
    Write-Host "  ✓ Require pull request reviews (1 approval)" -ForegroundColor White
    Write-Host "  ✓ Dismiss stale reviews when new commits pushed" -ForegroundColor White
}
Write-Host "  ✓ Require status checks to pass (version-check)" -ForegroundColor White
Write-Host "  ✓ Require branches to be up to date" -ForegroundColor White
Write-Host "  ✓ Prevent force pushes" -ForegroundColor White
Write-Host "  ✓ Prevent deletions" -ForegroundColor White
Write-Host "  ✓ Allow administrators to bypass" -ForegroundColor White
Write-Host ""

$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne 'y') {
    Write-Host "❌ Aborted" -ForegroundColor Red
    exit 0
}

Write-Host ""
Write-Host "🔧 Configuring branch protection..." -ForegroundColor Yellow

# Build the command
$protectionArgs = @(
    "api",
    "-X", "PUT",
    "/repos/$repo/branches/$branch/protection",
    "-f", "required_status_checks[strict]=true",
    "-f", "required_status_checks[contexts][]=version-check",
    "-f", "enforce_admins=false",
    "-f", "required_linear_history=false",
    "-f", "allow_force_pushes=false",
    "-f", "allow_deletions=false",
    "-f", "required_conversation_resolution=true"
)

if (-not $BypassReviews) {
    $protectionArgs += @(
        "-f", "required_pull_request_reviews[dismiss_stale_reviews]=true",
        "-f", "required_pull_request_reviews[require_code_owner_reviews]=false",
        "-f", "required_pull_request_reviews[required_approving_review_count]=1"
    )
}

# Apply protection
try {
    $result = & gh @protectionArgs 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to apply branch protection" -ForegroundColor Red
        Write-Host $result -ForegroundColor Red
        Write-Host ""
        Write-Host "Note: You need admin permissions on the repository" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "✅ Branch protection configured successfully!" -ForegroundColor Green
    Write-Host ""
    
    # Display current settings
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "   CURRENT PROTECTION STATUS" -ForegroundColor Cyan
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    gh api "/repos/$repo/branches/$branch/protection" | ConvertFrom-Json | Format-List
    
    Write-Host ""
    Write-Host "✅ Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "View in browser: https://github.com/$repo/settings/branches" -ForegroundColor Cyan
    Write-Host ""
    
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}

# Additional recommendations
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   RECOMMENDED NEXT STEPS" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Add CODEOWNERS file for automatic review assignments" -ForegroundColor White
Write-Host "   Create: .github/CODEOWNERS" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Configure additional status checks in workflows" -ForegroundColor White
Write-Host "   Edit: .github/workflows/*.yml" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Set up branch naming conventions" -ForegroundColor White
Write-Host "   Use: feature/*, bugfix/*, hotfix/*" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Review protection rules regularly" -ForegroundColor White
Write-Host "   Run: gh api /repos/$repo/branches/$branch/protection" -ForegroundColor Gray
Write-Host ""
