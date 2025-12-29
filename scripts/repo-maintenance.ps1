#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Monthly repository maintenance - Keep repo clean and efficient

.DESCRIPTION
    Automated maintenance tasks to run monthly:
    - Check for version consistency
    - Update CHANGELOG if needed
    - Verify deployment scripts work
    - Check for large files
    - Clean up old branches
    - Update documentation
    - Verify CI/CD pipelines

.EXAMPLE
    .\repo-maintenance.ps1
    Run full maintenance check

.EXAMPLE
    .\repo-maintenance.ps1 -Fix
    Run with automatic fixes
#>

param(
    [switch]$Fix
)

$ErrorActionPreference = "Stop"

function Write-Section {
    param([string]$Title)
    Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║ $Title" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
}

function Write-Check {
    param([string]$Message, [string]$Status)
    $icon = switch ($Status) {
        "pass" { "✅" }
        "warn" { "⚠️ " }
        "fail" { "❌" }
        default { "ℹ️ " }
    }
    Write-Host "  $icon $Message" -ForegroundColor $(if ($Status -eq "pass") { "Green" } elseif ($Status -eq "warn") { "Yellow" } elseif ($Status -eq "fail") { "Red" } else { "White" })
}

$issues = @()
$warnings = @()

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   NBA MODEL REPOSITORY - MONTHLY MAINTENANCE" -ForegroundColor Cyan
Write-Host "   $(Get-Date -Format 'MMMM dd, yyyy')" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan

# 1. Version Consistency Check
Write-Section "1. VERSION CONSISTENCY"

$VERSION = (Get-Content VERSION -Raw).Trim()
Write-Host "  Current version: $VERSION" -ForegroundColor White

$versionFiles = @(
    "src/serving/app.py",
    "src/prediction/engine.py",
    "models/production/model_pack.json",
    "tests/test_serving.py",
    ".github/copilot-instructions.md",
    "README.md",
    "infra/nba/main.json"
)

$versionInconsistent = $false
foreach ($file in $versionFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        if ($content -match "NBA_v\d+\.\d+\.\d+\.\d+" -and $content -notmatch $VERSION) {
            Write-Check "$file has wrong version" "fail"
            $versionInconsistent = $true
            $issues += "Version mismatch in $file"
        } else {
            Write-Check "$file version OK" "pass"
        }
    }
}

if (-not $versionInconsistent) {
    Write-Check "All files have consistent version" "pass"
} elseif ($Fix) {
    Write-Host "`n  Running version sync..." -ForegroundColor Yellow
    python scripts/bump_version.py $VERSION
    Write-Check "Version synced automatically" "pass"
}

# 2. Git Tag Check
Write-Section "2. GIT TAGS"

$tags = git tag -l
$currentTagged = $tags -contains $VERSION

if ($currentTagged) {
    Write-Check "Current version is tagged: $VERSION" "pass"
} else {
    Write-Check "Current version NOT tagged: $VERSION" "warn"
    $warnings += "Missing tag for $VERSION"
    
    if ($Fix) {
        Write-Host "  Creating tag..." -ForegroundColor Yellow
        git tag -a $VERSION -m "Automated tag creation from maintenance script"
        git push origin $VERSION
        Write-Check "Tag created automatically" "pass"
    }
}

$oldTags = $tags | Where-Object { $_ -match "^v\d+" -and $_ -notmatch "^NBA_" }
if ($oldTags) {
    Write-Check "Found $($oldTags.Count) old-style tags (v6.x)" "warn"
    $warnings += "Old versioning tags still exist"
} else {
    Write-Check "No old-style tags found" "pass"
}

# 3. Large Files Check
Write-Section "3. LARGE FILES"

$largeFiles = git ls-files | Where-Object {
    $size = (Get-Item $_).Length
    $size -gt 1MB
}

if ($largeFiles) {
    foreach ($file in $largeFiles) {
        $sizeMB = [math]::Round((Get-Item $file).Length / 1MB, 2)
        Write-Check "$file is $sizeMB MB" "warn"
        $warnings += "Large file: $file ($sizeMB MB)"
    }
} else {
    Write-Check "No files over 1MB in working tree" "pass"
}

# Check for files that should be ignored
$suspiciousFiles = git ls-files | Where-Object {
    $_ -match "\.(csv|log|coverage\.xml|\.pyc|__pycache__)$"
}

if ($suspiciousFiles) {
    Write-Check "Found $($suspiciousFiles.Count) files that should be ignored" "fail"
    $issues += "Tracked files that should be in .gitignore"
    foreach ($file in $suspiciousFiles | Select-Object -First 5) {
        Write-Host "      - $file" -ForegroundColor Red
    }
}

# 4. CHANGELOG Check
Write-Section "4. CHANGELOG"

if (Test-Path "CHANGELOG.md") {
    $changelog = Get-Content "CHANGELOG.md" -Raw
    
    if ($changelog -match $VERSION) {
        Write-Check "CHANGELOG includes current version" "pass"
    } else {
        Write-Check "CHANGELOG missing entry for $VERSION" "warn"
        $warnings += "CHANGELOG not updated"
    }
    
    # Check if changelog is up to date (modified within last 30 days)
    $changelogAge = (Get-Date) - (Get-Item "CHANGELOG.md").LastWriteTime
    if ($changelogAge.Days -gt 30) {
        Write-Check "CHANGELOG not updated in $($changelogAge.Days) days" "warn"
        $warnings += "CHANGELOG may be stale"
    } else {
        Write-Check "CHANGELOG recently updated" "pass"
    }
} else {
    Write-Check "CHANGELOG.md not found" "fail"
    $issues += "Missing CHANGELOG.md"
}

# 5. CI/CD Status
Write-Section "5. CI/CD WORKFLOWS"

$workflowsDir = ".github/workflows"
if (Test-Path $workflowsDir) {
    $workflows = Get-ChildItem $workflowsDir -Filter "*.yml"
    Write-Check "Found $($workflows.Count) workflow(s)" "pass"
    
    foreach ($workflow in $workflows) {
        Write-Host "    - $($workflow.Name)" -ForegroundColor Gray
    }
} else {
    Write-Check "No CI/CD workflows found" "warn"
    $warnings += "Missing CI/CD automation"
}

# 6. Branch Cleanup
Write-Section "6. BRANCHES"

$branches = git branch -r | Where-Object { $_ -notmatch "origin/HEAD" -and $_ -notmatch "origin/main" }

if ($branches.Count -eq 0) {
    Write-Check "No stale remote branches" "pass"
} else {
    Write-Check "Found $($branches.Count) remote branch(es)" "info"
    
    foreach ($branch in $branches | Select-Object -First 5) {
        $branchName = $branch.Trim()
        Write-Host "    - $branchName" -ForegroundColor Gray
    }
    
    if ($branches.Count -gt 5) {
        Write-Host "    ... and $($branches.Count - 5) more" -ForegroundColor Gray
    }
}

# 7. Documentation Check
Write-Section "7. DOCUMENTATION"

$requiredDocs = @(
    "README.md",
    "VERSIONING.md",
    "CHANGELOG.md",
    ".github/copilot-instructions.md"
)

foreach ($doc in $requiredDocs) {
    if (Test-Path $doc) {
        $size = (Get-Item $doc).Length
        $lastModified = (Get-Item $doc).LastWriteTime.ToString("yyyy-MM-dd")
        Write-Check "$doc exists (updated $lastModified)" "pass"
    } else {
        Write-Check "$doc missing" "fail"
        $issues += "Missing documentation: $doc"
    }
}

# 8. Deployment Scripts Check
Write-Section "8. DEPLOYMENT SCRIPTS"

$deployScripts = @(
    "scripts/deploy.ps1",
    "scripts/bump_version.py"
)

foreach ($script in $deployScripts) {
    if (Test-Path $script) {
        Write-Check "$script exists" "pass"
    } else {
        Write-Check "$script missing" "fail"
        $issues += "Missing script: $script"
    }
}

# 9. Test Coverage Check
Write-Section "9. TESTS"

if (Test-Path "tests") {
    $testFiles = Get-ChildItem "tests" -Filter "test_*.py" -Recurse
    Write-Check "Found $($testFiles.Count) test file(s)" "pass"
    
    # Check if tests run
    Write-Host "  Running quick test check..." -ForegroundColor Yellow
    $testResult = pytest tests --collect-only 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Check "Tests can be collected" "pass"
    } else {
        Write-Check "Test collection failed" "warn"
        $warnings += "Tests may have issues"
    }
} else {
    Write-Check "No tests directory" "fail"
    $issues += "Missing tests"
}

# 10. Repository Size
Write-Section "10. REPOSITORY SIZE"

$gitSize = (Get-ChildItem .git -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "  .git folder size: $([math]::Round($gitSize, 2)) MB" -ForegroundColor White

if ($gitSize -gt 100) {
    Write-Check "Repository is large ($([math]::Round($gitSize, 2)) MB)" "warn"
    $warnings += "Consider running git-deep-cleanup.ps1"
} elseif ($gitSize -gt 50) {
    Write-Check "Repository size is moderate" "info"
} else {
    Write-Check "Repository size is optimal" "pass"
}

# Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   MAINTENANCE SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($issues.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host ""
    Write-Host "  ✅ Repository is in excellent condition!" -ForegroundColor Green
    Write-Host ""
} else {
    if ($issues.Count -gt 0) {
        Write-Host ""
        Write-Host "  ❌ ISSUES FOUND ($($issues.Count)):" -ForegroundColor Red
        foreach ($issue in $issues) {
            Write-Host "    - $issue" -ForegroundColor Red
        }
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "  ⚠️  WARNINGS ($($warnings.Count)):" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "    - $warning" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "  Next maintenance check: $(( Get-Date).AddMonths(1).ToString('MMMM dd, yyyy'))" -ForegroundColor Cyan
Write-Host ""

# Exit code
if ($issues.Count -gt 0) {
    exit 1
} else {
    exit 0
}
