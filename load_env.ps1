# Load environment variables from .env file (PowerShell version)

if (-not (Test-Path .env)) {
    Write-Host "ERROR: .env file not found" -ForegroundColor Red
    Write-Host "Run: python scripts/setup_teams_webhook.py"
    exit 1
}

Write-Host "Loading environment variables from .env..." -ForegroundColor Cyan

Get-Content .env | ForEach-Object {
    # Skip comments and empty lines
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') {
        return
    }

    # Parse key=value
    if ($_ -match '^([^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim()

        # Set environment variable
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
        Write-Host "Set $key" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Environment variables loaded!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run:" -ForegroundColor Yellow
Write-Host "  python scripts/post_to_teams.py"
Write-Host ""
