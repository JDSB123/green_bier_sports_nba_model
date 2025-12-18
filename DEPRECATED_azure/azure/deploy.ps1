param(
    [string]$FunctionAppName,
    [string]$ResourceGroup
)

$ErrorActionPreference = "Stop"

Write-Host "Deploying Function App: $FunctionAppName"

# Check for Azure Functions Core Tools
if (-not (Get-Command func -ErrorAction SilentlyContinue)) {
    Write-Host "Azure Functions Core Tools not found." -ForegroundColor Red
    exit 1
}

# Change to function_app directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$functionAppDir = Join-Path $scriptPath "function_app"
Set-Location $functionAppDir

# Deploy
Write-Host "Publishing to Azure..."
func azure functionapp publish $FunctionAppName --python

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment complete!" -ForegroundColor Green
    Write-Host "URL: https://$FunctionAppName.azurewebsites.net"
} else {
    Write-Host "Deployment failed." -ForegroundColor Red
    exit 1
}

