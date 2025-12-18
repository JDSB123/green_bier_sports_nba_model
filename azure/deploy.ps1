# Azure Function App Deployment Script (PowerShell)
# Usage: .\azure\deploy.ps1 -FunctionAppName <name> -ResourceGroup <group>

param(
    [Parameter(Mandatory=$true)]
    [string]$FunctionAppName,
    
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup
)

Write-Host "🚀 Deploying Azure Function App: $FunctionAppName" -ForegroundColor Cyan

# Change to function app directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$functionAppDir = Join-Path $scriptPath "function_app"
Set-Location $functionAppDir

# Check for Azure Functions Core Tools
if (-not (Get-Command func -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Azure Functions Core Tools not found. Please install:" -ForegroundColor Red
    Write-Host "   npm install -g azure-functions-core-tools@4 --unsafe-perm true" -ForegroundColor Yellow
    exit 1
}

# Deploy function app
Write-Host "📦 Deploying to Azure..." -ForegroundColor Cyan
func azure functionapp publish $FunctionAppName --python

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Deployment complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Function endpoints:" -ForegroundColor Cyan
    Write-Host "  - Generate Picks: https://${FunctionAppName}.azurewebsites.net/api/generate_picks"
    Write-Host "  - Teams Bot: https://${FunctionAppName}.azurewebsites.net/api/teams/bot"
    Write-Host "  - Live Tracker: https://${FunctionAppName}.azurewebsites.net/api/live_tracker"
} else {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
}