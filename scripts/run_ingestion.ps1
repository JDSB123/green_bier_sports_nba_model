# Runs independent endpoint collectors in parallel and merges outputs
param(
    [string]$RepoRoot = (Get-Location).Path
)

# Activate venv
$venvActivate = Join-Path $RepoRoot ".venv/Scripts/Activate.ps1"
if (-not (Test-Path $venvActivate)) { throw "Missing .venv at $RepoRoot" }
. $venvActivate

# Ensure PYTHONPATH includes src
$env:PYTHONPATH = Join-Path $RepoRoot "src"

# Define collectors (add more as separate scripts)
$collectors = @(
    "python scripts/collect_the_odds.py"
    # "python scripts/collect_betsapi.py"  # add when available
)

# Start jobs in parallel
$jobs = @()
foreach ($cmd in $collectors) {
    $jobs += Start-Job -ScriptBlock { param($c) & pwsh -NoProfile -Command $c } -ArgumentList $cmd
}

# Wait for completion and report
$results = @()
foreach ($j in $jobs) {
    $res = Receive-Job -Job $j -Wait -AutoRemoveJob
    $results += $res
}

# Merge outputs
python scripts/merge_odds.py

Write-Host "Parallel collectors complete. Merged odds at data/processed/odds_merged.csv"