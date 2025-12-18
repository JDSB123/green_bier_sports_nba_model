# Check Function App Status

param(
    [string]$FunctionAppName = "green-bier-sports-nba",
    [string]$ResourceGroup = "green-bier-sport-ventures-rg"
)

Write-Host "Checking Function App Status..." -ForegroundColor Cyan
Write-Host ""

# Check if function app exists
Write-Host "1. Function App Status:" -ForegroundColor Yellow
$fa = az functionapp show --name $FunctionAppName --resource-group $ResourceGroup --query "{name:name, state:state, url:defaultHostName, runtime:siteConfig.linuxFxVersion}" -o json 2>&1 | ConvertFrom-Json
if ($fa) {
    Write-Host "   Name: $($fa.name)" -ForegroundColor Green
    Write-Host "   State: $($fa.state)" -ForegroundColor Green
    Write-Host "   URL: https://$($fa.url)" -ForegroundColor Green
    Write-Host "   Runtime: $($fa.runtime)" -ForegroundColor Green
} else {
    Write-Host "   [ERROR] Function App not found!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Check functions
Write-Host "2. Deployed Functions:" -ForegroundColor Yellow
$functions = az functionapp function list --name $FunctionAppName --resource-group $ResourceGroup --query "[].{name:name, language:language}" -o json 2>&1 | ConvertFrom-Json
if ($functions) {
    foreach ($func in $functions) {
        Write-Host "   - $($func.name) ($($func.language))" -ForegroundColor Green
    }
} else {
    Write-Host "   [WARN] No functions found! Deployment may have failed." -ForegroundColor Yellow
}
Write-Host ""

# Test endpoints
Write-Host "3. Testing Endpoints:" -ForegroundColor Yellow
$baseUrl = "https://$($fa.url)"

$endpoints = @(
    "/api/generate_picks?date=today",
    "/api/live_tracker?date=today"
)

foreach ($endpoint in $endpoints) {
    $url = $baseUrl + $endpoint
    Write-Host "   Testing: $url" -ForegroundColor Cyan
    try {
        $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        Write-Host "   [OK] Status: $($response.StatusCode)" -ForegroundColor Green
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "   [ERROR] Status: $statusCode - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
if (-not $functions) {
    Write-Host "   Functions are not deployed. Run: .\azure\DEPLOY_WITH_DEPS.ps1" -ForegroundColor Yellow
}
