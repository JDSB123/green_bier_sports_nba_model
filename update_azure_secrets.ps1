# PowerShell script to update Azure Key Vault secrets for NBA prediction system
# This ensures all API keys are properly stored in Azure Key Vault for production use

Write-Host "🔐 Updating Azure Key Vault Secrets for NBA Prediction System" -ForegroundColor Green
Write-Host "=" * 70

# Azure Key Vault name
$keyVaultName = "nbagbs-keyvault"

# API Keys to update
$secrets = @(
    @{
        Name = "THE-ODDS-API-KEY"
        Value = "4a0b80471d1ebeeb74c358fa0fcc4a27"
        Description = "The Odds API - Active subscription key"
    },
    @{
        Name = "API-BASKETBALL-KEY"
        Value = "eea8757fae3c507add2df14800bae25f"
        Description = "API-Basketball - Working key for team stats and game data"
    }
)

# Check Azure CLI login
try {
    $account = az account show 2>$null | ConvertFrom-Json
    Write-Host "✅ Azure CLI logged in as: $($account.user.name)" -ForegroundColor Green
    Write-Host "   Subscription: $($account.name)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Azure CLI not logged in. Please run 'az login' first." -ForegroundColor Red
    exit 1
}

# Update each secret in Key Vault
foreach ($secret in $secrets) {
    Write-Host "🔄 Updating $($secret.Name)..." -ForegroundColor Cyan

    try {
        # Set the secret in Key Vault
        az keyvault secret set --vault-name $keyVaultName --name $secret.Name --value $secret.Value --description $secret.Description 2>$null | Out-Null

        if ($LASTEXITCODE -eq 0) {
            Write-Host "   ✅ Successfully updated $($secret.Name)" -ForegroundColor Green
        } else {
            Write-Host "   ❌ Failed to update $($secret.Name)" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ❌ Error updating $($secret.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Verify secrets were set correctly
Write-Host "`n🔍 Verifying Key Vault secrets..." -ForegroundColor Cyan

foreach ($secret in $secrets) {
    try {
        $result = az keyvault secret show --vault-name $keyVaultName --name $secret.Name --query "value" -o tsv 2>$null

        if ($result -and $result.Trim() -eq $secret.Value) {
            Write-Host "   ✅ $($secret.Name): Set correctly" -ForegroundColor Green
        } else {
            Write-Host "   ❌ $($secret.Name): Verification failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "   ❌ $($secret.Name): Could not verify" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Azure Key Vault secrets update complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Redeploy Azure Container App to pick up new secrets"
Write-Host "2. Test prediction endpoint with new API keys"
Write-Host "3. Verify feature engineering works with full API access"
Write-Host "`nCommands for redeployment:" -ForegroundColor Yellow
Write-Host "  az containerapp up --name nba-gbsv-api --resource-group nba-gbsv-model-rg --source ."
Write-Host "`nOr update existing deployment:" -ForegroundColor Yellow
Write-Host "  az deployment group create -g nba-gbsv-model-rg -f infra/nba/main.bicep --parameters theOddsApiKey='...' apiBasketballKey='...'"