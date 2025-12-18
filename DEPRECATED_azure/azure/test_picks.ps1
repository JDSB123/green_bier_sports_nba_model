# Quick Test Script for NBA Picks
# Usage: .\azure\test_picks.ps1 [-Date today] [-Matchup Lakers] [-PostToTeams]

param(
    [Parameter(Mandatory=$false)]
    [string]$FunctionAppName = "green-bier-sports-nba",
    
    [Parameter(Mandatory=$false)]
    [string]$Date = "today",
    
    [Parameter(Mandatory=$false)]
    [string]$Matchup = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$PostToTeams = $false
)

$ErrorActionPreference = "Stop"

# Your Teams channel ID
$TeamsChannelId = "19:5369a11408864936935266147c1f3b02@thread.tacv2"

$functionUrl = "https://${FunctionAppName}.azurewebsites.net/api/generate_picks"

Write-Host "🏀 Testing NBA Picks Generation" -ForegroundColor Cyan
Write-Host ""

# Build query string
$queryParams = @("date=$Date")
if (-not [string]::IsNullOrEmpty($Matchup)) {
    $queryParams += "matchup=$Matchup"
}
if ($PostToTeams) {
    $queryParams += "channel_id=$TeamsChannelId"
}

$fullUrl = "$functionUrl?" + ($queryParams -join "&")

Write-Host "📡 Calling: $fullUrl" -ForegroundColor Yellow
Write-Host ""

try {
    $result = Invoke-RestMethod -Uri $fullUrl -Method Get -TimeoutSec 60
    
    Write-Host "✅ SUCCESS!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Results:" -ForegroundColor Cyan
    Write-Host "   Date: $($result.date)" -ForegroundColor White
    Write-Host "   Total Plays: $($result.total_plays)" -ForegroundColor White
    Write-Host "   Games: $($result.games)" -ForegroundColor White
    Write-Host ""
    
    if ($result.predictions) {
        Write-Host "🎯 Picks:" -ForegroundColor Cyan
        foreach ($pred in $result.predictions) {
            Write-Host "   $($pred.matchup)" -ForegroundColor Yellow
            foreach ($play in $pred.plays) {
                $fireEmoji = "🔥" * [Math]::Min(5, [Math]::Max(1, [int]($play.confidence * 5)))
                Write-Host "      • $($play.period) $($play.market): $($play.pick) | Edge: $($play.edge) | Conf: $([Math]::Round($play.confidence * 100))% $fireEmoji" -ForegroundColor White
            }
            Write-Host ""
        }
    }
    
    if ($PostToTeams) {
        Write-Host "✅ Posted to Teams channel!" -ForegroundColor Green
    }
    
    # Save to file
    $outputFile = "picks_$($result.date).json"
    $result | ConvertTo-Json -Depth 10 | Out-File $outputFile
    Write-Host "💾 Saved to: $outputFile" -ForegroundColor Cyan
    
} catch {
    Write-Host "❌ ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "   Response: $responseBody" -ForegroundColor Gray
    }
    exit 1
}