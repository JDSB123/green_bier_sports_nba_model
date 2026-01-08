#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Initialize the nba-historical-data git repository structure.
    
.DESCRIPTION
    Creates a new git repository for storing historical NBA data ingestion,
    backtesting models, and archived picks with timestamps/versions.
    
    This repository is separate from the main NBA model repository and is
    designed to store historical data that should be preserved rather than
    ignored in git.
    
.PARAMETER RepoPath
    Path where the new repository should be created. Defaults to parent directory.
    
.EXAMPLE
    .\scripts\init_historical_data_repo.ps1
    
.EXAMPLE
    .\scripts\init_historical_data_repo.ps1 -RepoPath "C:\repos\nba-historical-data"
#>

param(
    [string]$RepoPath = (Join-Path (Split-Path $PSScriptRoot -Parent -Resolve) ".." "nba-historical-data")
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== Initializing NBA Historical Data Repository ===" -ForegroundColor Cyan
Write-Host "Repository path: $RepoPath`n" -ForegroundColor Gray

# Resolve absolute path
$RepoPath = Resolve-Path $RepoPath -ErrorAction SilentlyContinue
if (-not $RepoPath) {
    $RepoPath = (New-Item -ItemType Directory -Path $RepoPath -Force).FullName
}

# Create directory structure
$directories = @(
    "data",
    "data/raw",
    "data/processed",
    "data/historical",
    "data/historical/the_odds",
    "data/historical/the_odds/events",
    "data/historical/the_odds/odds",
    "data/historical/the_odds/period_odds",
    "data/historical/the_odds/player_props",
    "data/historical/the_odds/metadata",
    "data/historical/exports",
    "data/archived_picks",
    "data/archived_picks/by_date",
    "data/archived_picks/by_version",
    "models",
    "models/backtest",
    "models/fine_tuned",
    "backtest_results",
    "backtest_results/by_season",
    "backtest_results/by_market",
    "scripts",
    "docs",
    ".github",
    ".github/workflows"
)

Write-Host "Creating directory structure..." -ForegroundColor Yellow
foreach ($dir in $directories) {
    $fullPath = Join-Path $RepoPath $dir
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    Write-Host "  ✓ $dir" -ForegroundColor Gray
}

# Create README.md
$readmeContent = @"
# NBA Historical Data Repository

This repository stores historical NBA data ingestion, backtesting models, and archived picks for fine-tuning and production improvements.

## Repository Structure

\`\`\`
nba-historical-data/
├── data/
│   ├── raw/                    # Raw ingested data (before processing)
│   ├── processed/              # Processed/cleaned data
│   ├── historical/            # Historical odds and events
│   │   ├── the_odds/          # Raw JSON from The Odds API
│   │   │   ├── events/        # Historical events by season
│   │   │   ├── odds/          # Historical odds snapshots
│   │   │   ├── period_odds/   # Historical period odds
│   │   │   ├── player_props/  # Player props (if available)
│   │   │   └── metadata/      # Ingestion tracking
│   │   └── exports/           # Normalized CSV/Parquet exports
│   └── archived_picks/        # Archived picks with timestamps/versions
│       ├── by_date/           # Organized by date
│       └── by_version/        # Organized by model version
├── models/
│   ├── backtest/              # Backtesting model artifacts
│   └── fine_tuned/            # Fine-tuned model versions
├── backtest_results/          # Backtest analysis results
│   ├── by_season/             # Results organized by season
│   └── by_market/             # Results organized by market
├── scripts/                   # Utility scripts for data management
└── docs/                      # Documentation
\`\`\`

## Purpose

1. **Historical Data Preservation**: Store all historical data ingestion (not ignored in git)
2. **Backtesting Models**: Archive backtesting model artifacts for reproducibility
3. **Pick Archival**: Store picks with unique timestamps/versions before front-end overwrites
4. **Fine-tuning Preparation**: Maintain historical data for model fine-tuning

## Azure Blob Storage

Historical data is also synced to Azure Blob Storage:
- **Container**: `nbahistoricaldata`
- **Resource Group**: `nba-gbsv-model-rg`
- **Storage Account**: `nbagbsvstrg`

Data is organized with unique timestamps and versions to prevent overwrites.

## Usage

### Archiving Picks

\`\`\`powershell
# Archive picks before front-end update
.\scripts\archive_picks_to_azure.ps1 -Date "2025-01-15"
\`\`\`

### Syncing Historical Data

\`\`\`powershell
# Sync historical data to Azure
.\scripts\sync_historical_data_to_azure.ps1 -Season "2024-2025"
\`\`\`

### Backtest Model Storage

\`\`\`powershell
# Store backtest model artifacts
.\scripts\store_backtest_model.ps1 -Version "NBA_v33.0.11.0" -BacktestDate "2025-01-15"
\`\`\`

## Data Organization

- **Timestamps**: All data includes ISO 8601 timestamps
- **Versions**: Model versions are tracked (e.g., NBA_v33.0.11.0)
- **Unique Identifiers**: Prevents overwrites with versioned paths

## Git Strategy

Unlike the main NBA model repository, this repository:
- ✅ **Commits** historical data (not ignored)
- ✅ **Tracks** all ingested data
- ✅ **Preserves** backtest models
- ✅ **Archives** picks with full history

## Related Repositories

- **Main NBA Model**: \`green_bier_sports_nba_model\` (production model)
- **Historical Data**: \`nba-historical-data\` (this repository)

## Contributing

When adding historical data:
1. Ensure data includes timestamp and version metadata
2. Organize by date/season/market as appropriate
3. Update documentation for new data structures
4. Sync to Azure blob storage for redundancy
"@

$readmePath = Join-Path $RepoPath "README.md"
Set-Content -Path $readmePath -Value $readmeContent -Encoding UTF8
Write-Host "`n✓ Created README.md" -ForegroundColor Green

# Create .gitignore
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/
*.whl

# Virtual environments
.venv/
venv/
ENV/
env/

# Environment variables
.env
.env.*
!.env.example

# IDE
.vscode/*
!.vscode/settings.json
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Type checking
.mypy_cache/

# Temporary files
*.tmp
temp/

# Azure credentials (keep examples)
*.json
!*.example.json
!package.json
!tsconfig.json
"@

$gitignorePath = Join-Path $RepoPath ".gitignore"
Set-Content -Path $gitignorePath -Value $gitignoreContent -Encoding UTF8
Write-Host "✓ Created .gitignore" -ForegroundColor Green

# Create .gitkeep files for empty directories
$gitkeepDirs = @(
    "data/raw",
    "data/processed",
    "data/historical/the_odds/events",
    "data/historical/the_odds/odds",
    "data/historical/the_odds/period_odds",
    "data/historical/the_odds/player_props",
    "data/historical/the_odds/metadata",
    "data/historical/exports",
    "data/archived_picks/by_date",
    "data/archived_picks/by_version",
    "models/backtest",
    "models/fine_tuned",
    "backtest_results/by_season",
    "backtest_results/by_market"
)

Write-Host "`nCreating .gitkeep files..." -ForegroundColor Yellow
foreach ($dir in $gitkeepDirs) {
    $gitkeepPath = Join-Path $RepoPath $dir ".gitkeep"
    New-Item -ItemType File -Path $gitkeepPath -Force | Out-Null
    Write-Host "  ✓ $dir/.gitkeep" -ForegroundColor Gray
}

# Initialize git repository if not already initialized
Write-Host "`nInitializing git repository..." -ForegroundColor Yellow
Push-Location $RepoPath
try {
    if (-not (Test-Path ".git")) {
        git init | Out-Null
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
        
        # Create initial commit
        git add .
        git commit -m "Initial commit: NBA historical data repository structure" | Out-Null
        Write-Host "✓ Initial commit created" -ForegroundColor Green
    } else {
        Write-Host "⚠ Git repository already exists" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠ Could not initialize git repository: $_" -ForegroundColor Yellow
} finally {
    Pop-Location
}

Write-Host "`n=== Repository Initialization Complete ===" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Navigate to: $RepoPath" -ForegroundColor Gray
Write-Host "  2. Add remote: git remote add origin <your-repo-url>" -ForegroundColor Gray
Write-Host "  3. Push: git push -u origin main" -ForegroundColor Gray
Write-Host "`n" -ForegroundColor Gray
