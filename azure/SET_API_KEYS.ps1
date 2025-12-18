# Set API Keys from .env file
param(
    [string]$FunctionAppName = "green-bier-sports-nba",
    [string]$ResourceGroup = "green-bier-sport-ventures-rg"
)

$envFile = Join-Path $PSScriptRoot "..\.env"
$theOddsKey = ""
$apiBasketballKey = ""

if (Test-Path $envFile) {
    Write-Host "Reading .env file..." -ForegroundColor Cyan
    $content = Get-Content $envFile
    foreach ($line in $content) {
        $line = $line.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $idx = $line.IndexOf("=")
            $key = $line.Substring(0, $idx).Trim()
            $value = $line.Substring($idx + 1).Trim()
            if ($key -eq "THE_ODDS_API_KEY") { $theOddsKey = $value }
            if ($key -eq "API_BASKETBALL_KEY") { $apiBasketballKey = $value }
        }
    }
}

$settings = @("TEAMS_TENANT_ID=18ee0910-417d-4a81-a3f5-7945bdbd5a78")
if ($theOddsKey) { 
    $settings += "THE_ODDS_API_KEY=$theOddsKey"
    Write-Host "Found THE_ODDS_API_KEY" -ForegroundColor Green
}
if ($apiBasketballKey) { 
    $settings += "API_BASKETBALL_KEY=$apiBasketballKey"
    Write-Host "Found API_BASKETBALL_KEY" -ForegroundColor Green
}

Write-Host "Setting app settings..." -ForegroundColor Cyan
az functionapp config appsettings set `
    --name $FunctionAppName `
    --resource-group $ResourceGroup `
    --settings $settings

Write-Host "Done!" -ForegroundColor Green