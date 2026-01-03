# PowerShell script to set up the complete environment for NBA prediction model
# This script retrieves secrets from Azure Key Vault and configures the environment

Write-Host "Setting up NBA Prediction Model Environment" -ForegroundColor Green
Write-Host "================================================"

# Set working directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if Azure CLI is available
try {
    $azVersion = az --version 2>$null
    Write-Host "Azure CLI available" -ForegroundColor Green
} catch {
    Write-Host "Azure CLI not found. Please install Azure CLI first." -ForegroundColor Red
    Write-Host "Download from: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli" -ForegroundColor Yellow
    exit 1
}

# Function to get secret from Azure Key Vault
function Get-KeyVaultSecret {
    param (
        [string]$secretName,
        [string]$vaultName = "nbagbs-keyvault"
    )

    try {
        $secretValue = az keyvault secret show --vault-name $vaultName --name $secretName --query value -o tsv 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $secretValue.Trim()
        } else {
            Write-Host "Secret '$secretName' not found in Key Vault '$vaultName'" -ForegroundColor Yellow
            return $null
        }
    } catch {
        Write-Host "Failed to retrieve secret '$secretName' from Key Vault" -ForegroundColor Red
        return $null
    }
}

# Create .env file if it doesn't exist
$envFile = ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "Creating .env file from template..."
    Copy-Item ".env.example" ".env" -Force
}

# Read existing secrets from secrets directory
Write-Host "Reading existing secrets from secrets/ directory..."

$secretsDir = "secrets"
$theOddsKey = $null
$apiBasketballKey = $null
$actionNetworkUsername = $null
$actionNetworkPassword = $null

if (Test-Path "$secretsDir/THE_ODDS_API_KEY") {
    $theOddsKey = Get-Content "$secretsDir/THE_ODDS_API_KEY" -Raw
    Write-Host "Found THE_ODDS_API_KEY"
}

if (Test-Path "$secretsDir/API_BASKETBALL_KEY") {
    $apiBasketballKey = Get-Content "$secretsDir/API_BASKETBALL_KEY" -Raw
    Write-Host "Found API_BASKETBALL_KEY"
}

if (Test-Path "$secretsDir/ACTION_NETWORK_USERNAME") {
    $actionNetworkUsername = Get-Content "$secretsDir/ACTION_NETWORK_USERNAME" -Raw
    Write-Host "Found ACTION_NETWORK_USERNAME"
}

if (Test-Path "$secretsDir/ACTION_NETWORK_PASSWORD") {
    $actionNetworkPassword = Get-Content "$secretsDir/ACTION_NETWORK_PASSWORD" -Raw
    Write-Host "Found ACTION_NETWORK_PASSWORD"
}

# Try to get missing secrets from Azure Key Vault
Write-Host "Retrieving missing secrets from Azure Key Vault..."

if (-not $theOddsKey) {
    $theOddsKey = Get-KeyVaultSecret -secretName "THE-ODDS-API-KEY"
    if ($theOddsKey) {
        $theOddsKey | Out-File "$secretsDir/THE_ODDS_API_KEY" -Encoding UTF8 -NoNewline
        Write-Host "Retrieved THE_ODDS_API_KEY from Key Vault"
    }
}

if (-not $apiBasketballKey) {
    $apiBasketballKey = Get-KeyVaultSecret -secretName "API-BASKETBALL-KEY"
    if ($apiBasketballKey) {
        $apiBasketballKey | Out-File "$secretsDir/API_BASKETBALL_KEY" -Encoding UTF8 -NoNewline
        Write-Host "Retrieved API_BASKETBALL_KEY from Key Vault"
    }
}

# Update .env file with retrieved secrets
Write-Host "Updating .env file..."

$envContent = Get-Content ".env" -Raw

# Replace environment variables
if ($theOddsKey) {
    $envContent = $envContent -replace 'THE_ODDS_API_KEY=.*', "THE_ODDS_API_KEY=$theOddsKey"
}
if ($apiBasketballKey) {
    $envContent = $envContent -replace 'API_BASKETBALL_KEY=.*', "API_BASKETBALL_KEY=$apiBasketballKey"
}
if ($actionNetworkUsername) {
    $envContent = $envContent -replace 'ACTION_NETWORK_USERNAME=.*', "ACTION_NETWORK_USERNAME=$actionNetworkUsername"
}
if ($actionNetworkPassword) {
    $envContent = $envContent -replace 'ACTION_NETWORK_PASSWORD=.*', "ACTION_NETWORK_PASSWORD=$actionNetworkPassword"
}

# Set strict feature mode
$envContent = $envContent -replace 'PREDICTION_FEATURE_MODE=.*', 'PREDICTION_FEATURE_MODE=strict'

Set-Content ".env" $envContent -NoNewline

# Test the configuration
Write-Host "Testing configuration..."

try {
    # Test secret reading
    $testResult = python -c "
from src.config import settings
print('THE_ODDS_API_KEY:', bool(settings.the_odds_api_key))
print('API_BASKETBALL_KEY:', bool(settings.api_basketball_key))
print('ACTION_NETWORK_USERNAME:', bool(settings.action_network_username))
print('ACTION_NETWORK_PASSWORD:', bool(settings.action_network_password))
" 2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Configuration test passed!" -ForegroundColor Green
        Write-Host $testResult
    } else {
        Write-Host "Configuration test had issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not run configuration test" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Verify .env file has correct API keys"
Write-Host "2. Run: python scripts/predict.py --date 2025-01-03"
Write-Host "3. Check that predictions now use all 29 features"
Write-Host ""
Write-Host "If you encounter issues:" -ForegroundColor Yellow
Write-Host "- Check Azure Key Vault access: az login"
Write-Host "- Verify Key Vault name: nbagbs-keyvault"
Write-Host "- Update secret names if they have changed"