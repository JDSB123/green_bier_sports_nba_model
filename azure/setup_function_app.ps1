# Setup Azure Function App Infrastructure
# This script creates the Function App and required resources
# Usage: .\azure\setup_function_app.ps1 -FunctionAppName <name> -ResourceGroup <group> -Location <location>

param(
    [Parameter(Mandatory=$true)]
    [string]$FunctionAppName,
    
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup,
    
    [Parameter(Mandatory=$false)]
    [string]$Location = "eastus",
    
    [Parameter(Mandatory=$false)]
    [string]$StorageAccountName = "",
    
    [Parameter(Mandatory=$false)]
    [string]$AppServicePlanName = "",
    
    [Parameter(Mandatory=$false)]
    [string]$ContainerRegistryName = ""
)

$ErrorActionPreference = "Stop"

Write-Host "🚀 Setting up Azure Function App: $FunctionAppName" -ForegroundColor Cyan

# Generate storage account name if not provided
if ([string]::IsNullOrEmpty($StorageAccountName)) {
    $StorageAccountName = "$($FunctionAppName.Replace('-', ''))storage".Substring(0, [Math]::Min(24, "$($FunctionAppName.Replace('-', ''))storage".Length))
}

# Generate App Service Plan name if not provided
if ([string]::IsNullOrEmpty($AppServicePlanName)) {
    $AppServicePlanName = "$FunctionAppName-plan"
}

Write-Host "📋 Configuration:" -ForegroundColor Cyan
Write-Host "  Resource Group: $ResourceGroup"
Write-Host "  Location: $Location"
Write-Host "  Storage Account: $StorageAccountName"
Write-Host "  App Service Plan: $AppServicePlanName"

# Check if logged in to Azure
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "❌ Not logged in to Azure. Please run: az login" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green

# Create resource group
Write-Host "📦 Creating resource group..." -ForegroundColor Cyan
az group create --name $ResourceGroup --location $Location | Out-Null

# Create storage account
Write-Host "📦 Creating storage account..." -ForegroundColor Cyan
az storage account create `
    --name $StorageAccountName `
    --resource-group $ResourceGroup `
    --location $Location `
    --sku Standard_LRS | Out-Null

# Get storage connection string
$storageConnectionString = az storage account show-connection-string `
    --name $StorageAccountName `
    --resource-group $ResourceGroup `
    --query connectionString `
    --output tsv

# Create App Service Plan (Linux, Consumption plan for cost efficiency)
Write-Host "📦 Creating App Service Plan..." -ForegroundColor Cyan
az functionapp plan create `
    --name $AppServicePlanName `
    --resource-group $ResourceGroup `
    --location $Location `
    --sku Y1 `
    --is-linux | Out-Null

# Create Function App
Write-Host "📦 Creating Function App..." -ForegroundColor Cyan
az functionapp create `
    --name $FunctionAppName `
    --resource-group $ResourceGroup `
    --storage-account $StorageAccountName `
    --plan $AppServicePlanName `
    --runtime python `
    --runtime-version 3.11 `
    --functions-version 4 `
    --os-type Linux | Out-Null

# Configure app settings
Write-Host "⚙️  Configuring app settings..." -ForegroundColor Cyan

# Note: You'll need to set these manually or via Azure Key Vault
Write-Host "⚠️  IMPORTANT: Set the following app settings:" -ForegroundColor Yellow
Write-Host "  - THE_ODDS_API_KEY"
Write-Host "  - API_BASKETBALL_KEY"
Write-Host "  - ACTION_NETWORK_USERNAME (optional)"
Write-Host "  - ACTION_NETWORK_PASSWORD (optional)"
Write-Host ""
Write-Host "You can set them with:" -ForegroundColor Cyan
Write-Host "  az functionapp config appsettings set --name $FunctionAppName --resource-group $ResourceGroup --settings THE_ODDS_API_KEY='your-key' API_BASKETBALL_KEY='your-key'" -ForegroundColor Gray

# Enable Application Insights (optional but recommended)
Write-Host "📊 Enabling Application Insights..." -ForegroundColor Cyan
$aiName = "$FunctionAppName-ai"
az monitor app-insights component create `
    --app $aiName `
    --location $Location `
    --resource-group $ResourceGroup | Out-Null

$aiKey = az monitor app-insights component show `
    --app $aiName `
    --resource-group $ResourceGroup `
    --query instrumentationKey `
    --output tsv

az functionapp config appsettings set `
    --name $FunctionAppName `
    --resource-group $ResourceGroup `
    --settings APPINSIGHTS_INSTRUMENTATIONKEY=$aiKey | Out-Null

Write-Host "✅ Function App setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Set API keys in App Settings (see above)"
Write-Host "  2. Deploy function code: .\azure\deploy.ps1 -FunctionAppName $FunctionAppName -ResourceGroup $ResourceGroup"
Write-Host "  3. Configure Teams Bot (see README.md)"
Write-Host ""
Write-Host "Function App URL: https://$FunctionAppName.azurewebsites.net" -ForegroundColor Green