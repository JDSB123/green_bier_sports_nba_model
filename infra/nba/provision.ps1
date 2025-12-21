Param(
  [string]$Subscription = "",
  [string]$ResourceGroup = "greenbier-enterprise-rg",
  [string]$EnvironmentName = "greenbier-nba-env",
  [string]$AppName = "nba-picks-api",
  [string]$AcrName = "greenbieracr",
  [string]$Image = "greenbieracr.azurecr.io/nba-model:v6.4",
  [string]$Location = "eastus",
  [string]$KeyVaultName = "greenbier-keyvault"
)

$ErrorActionPreference = "Stop"

if ($Subscription -and $Subscription.Trim() -ne "") {
  az account set --subscription $Subscription
}

Write-Host "Ensuring Container Apps environment '$EnvironmentName' in $Location..."
az containerapp env show -n $EnvironmentName -g $ResourceGroup --only-show-errors 2>$null
if ($LASTEXITCODE -ne 0) {
  az containerapp env create -n $EnvironmentName -g $ResourceGroup -l $Location
}

Write-Host "Ensuring Container App '$AppName'..."
az containerapp show -n $AppName -g $ResourceGroup --only-show-errors 2>$null
if ($LASTEXITCODE -ne 0) {
  az containerapp create -n $AppName -g $ResourceGroup --environment $EnvironmentName --image $Image --registry-server "$AcrName.azurecr.io" --registry-identity system --ingress external --target-port 80
}

Write-Host "Assigning system identity to app..."
az containerapp identity assign -n $AppName -g $ResourceGroup --system-assigned
$principalId = az containerapp show -n $AppName -g $ResourceGroup --query identity.principalId -o tsv

Write-Host "Granting AcrPull to app identity..."
$subId = az account show --query id -o tsv
$acrScope = "/subscriptions/$subId/resourceGroups/$ResourceGroup/providers/Microsoft.ContainerRegistry/registries/$AcrName"
az role assignment create --assignee $principalId --scope $acrScope --role "AcrPull" 2>$null | Out-Null

Write-Host "Fetching secret values from Key Vault '$KeyVaultName'..."
$kvOddsVal = az keyvault secret show --vault-name $KeyVaultName --name THE-ODDS-API-KEY --query value -o tsv
if (-not $kvOddsVal) { Write-Error "Secret THE-ODDS-API-KEY not found or empty in Key Vault $KeyVaultName"; exit 1 }

$kvBasketVal = az keyvault secret show --vault-name $KeyVaultName --name API-BASKETBALL-KEY --query value -o tsv
if (-not $kvBasketVal) { Write-Error "Secret API-BASKETBALL-KEY not found or empty in Key Vault $KeyVaultName"; exit 1 }

Write-Host "Setting app secrets and environment variables..."
az containerapp secret set -n $AppName -g $ResourceGroup --secrets the-odds-api-key=$kvOddsVal api-basketball-key=$kvBasketVal
az containerapp update -n $AppName -g $ResourceGroup --set-env-vars THE_ODDS_API_KEY=secretref:the-odds-api-key API_BASKETBALL_KEY=secretref:api-basketball-key

Write-Host "Updating app image..."
az containerapp update -n $AppName -g $ResourceGroup --image $Image

$fqdn = az containerapp show -n $AppName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host "FQDN: $fqdn"
if ($fqdn) { curl -s "https://$fqdn/health" }
