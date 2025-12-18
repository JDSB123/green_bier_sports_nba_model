# Open Live Tracker in Browser
# Usage: .\azure\OPEN_LIVE_TRACKER.ps1 [-Date today]

param(
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "green-bier-sports-nba",
    
    [Parameter(Mandatory=$false)]
    [string]$Date = "today"
)

$trackerUrl = "https://${FunctionAppName}.azurewebsites.net/api/live_tracker?date=$Date"

Write-Host "🌐 Opening Live Tracker..." -ForegroundColor Cyan
Write-Host "   URL: $trackerUrl" -ForegroundColor White
Write-Host ""

Start-Process $trackerUrl

Write-Host "✅ Live tracker opened in your default browser!" -ForegroundColor Green
Write-Host "   The tracker will auto-refresh every 60 seconds." -ForegroundColor Gray