Param(
  [string]$Subscription = "",
  [string]$ResourceGroup = "greenbier-enterprise-rg",
  [string]$EnvironmentName = "greenbier-env",
  [string]$AppName = "nba-picks-api",
  [string]$AcrName = "greenbieracr",
  [string]$Image = "greenbieracr.azurecr.io/nba-model:v6.1",
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
$kvBasketVal = az keyvault secret show --vault-name $KeyVaultName --name API-BASKETBALL-KEY --query value -o tsv

Write-Host "Setting app secrets and environment variables..."
az containerapp secret set -n $AppName -g $ResourceGroup --secrets THE_ODDS_API_KEY=$kvOddsVal API_BASKETBALL_KEY=$kvBasketVal
az containerapp update -n $AppName -g $ResourceGroup --set-env-vars THE_ODDS_API_KEY=secretref:THE_ODDS_API_KEY API_BASKETBALL_KEY=secretref:API_BASKETBALL_KEY

Write-Host "Updating app image..."
az containerapp update -n $AppName -g $ResourceGroup --image $Image

$fqdn = az containerapp show -n $AppName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host "FQDN: $fqdn"
if ($fqdn) { curl -s "https://$fqdn/health" }
