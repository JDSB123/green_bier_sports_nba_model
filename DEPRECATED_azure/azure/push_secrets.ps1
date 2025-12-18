# Script to read .env and push to Azure Function App Settings
param(
    [string]$FunctionAppName,
    [string]$ResourceGroup
)

$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "❌ .env file not found" -ForegroundColor Red
    exit 1
}

Write-Host "reading .env file..."
$settings = @()
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Length -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            # Remove quotes if present
            if ($value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($key -in "THE_ODDS_API_KEY", "API_BASKETBALL_KEY", "ACTION_NETWORK_USERNAME", "ACTION_NETWORK_PASSWORD", "TEAMS_WEBHOOK_URL") {
                $settings += "$key=$value"
            }
        }
    }
}

if ($settings.Count -gt 0) {
    Write-Host "Pushing $($settings.Count) settings to Azure..."
    az functionapp config appsettings set --name $FunctionAppName --resource-group $ResourceGroup --settings $settings
} else {
    Write-Host "No relevant keys found in .env"
}

