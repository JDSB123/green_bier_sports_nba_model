Param(
  [string]$Subscription = "",
  [string]$SourceResourceGroup = "greenbier-enterprise-rg",
  [string]$DestResourceGroup = "nba-gbsv-model-rg",
  [string]$Location = "eastus",
  [string]$Environment = "prod",
  [string]$SourceAcrName = "greenbieracr",
  [string]$DestAcrName = "nbagbsacr",
  [string]$SourceKeyVaultName = "greenbier-keyvault",
  [string]$DestKeyVaultName = "nbagbs-keyvault",
  [string]$ContainerAppEnvName = "greenbier-nba-env",
  [string]$ContainerAppName = "nba-gbsv-api",
  [string]$ImageName = "nba-model",
  [string]$Tag = "v6.4"
)

$ErrorActionPreference = "Stop"

if ($Subscription -and $Subscription.Trim() -ne "") {
  az account set --subscription $Subscription
}

Write-Host "[1/6] Ensure destination RG: $DestResourceGroup" -ForegroundColor Yellow
az group create --name $DestResourceGroup --location $Location --output none

Write-Host "[2/6] Deploy shared stack to $DestResourceGroup (ACR/KV/Env)" -ForegroundColor Yellow
$sharedDeploy = az deployment group create `
  --resource-group $DestResourceGroup `
  --name "gbs-shared-$((Get-Date).ToString('yyyyMMdd-HHmmss'))" `
  --template-file "$(Split-Path -Parent $MyInvocation.MyCommand.Path)\..\shared\main.bicep" `
  --parameters environment=$Environment location=$Location containerRegistryName=$DestAcrName keyVaultName=$DestKeyVaultName containerAppEnvName=$ContainerAppEnvName `
  --output json | ConvertFrom-Json

if ($LASTEXITCODE -ne 0) { Write-Error "Shared deployment failed"; exit 1 }

$destAcrServer = $sharedDeploy.properties.outputs.containerRegistryLoginServer.value
$destKvName = $sharedDeploy.properties.outputs.keyVaultName.value
$envName = $sharedDeploy.properties.outputs.containerAppEnvironmentName.value

Write-Host "  ✓ Dest ACR: $destAcrServer" -ForegroundColor Green
Write-Host "  ✓ Dest KeyVault: $destKvName" -ForegroundColor Green
Write-Host "  ✓ Dest Env: $envName" -ForegroundColor Green

Write-Host "[3/6] Import images from source ACR to destination ACR" -ForegroundColor Yellow
$sourceAcrServer = "$SourceAcrName.azurecr.io"
az acr import -n $DestAcrName --source "${sourceAcrServer}/${ImageName}:${Tag}" --image "${ImageName}:${Tag}" --output none
az acr import -n $DestAcrName --source "${sourceAcrServer}/${ImageName}:latest" --image "${ImageName}:latest" --output none 2>$null
Write-Host "  ✓ Imported ${ImageName}:${Tag} and latest" -ForegroundColor Green

Write-Host "[4/6] Clone secrets from source Key Vault to destination" -ForegroundColor Yellow
$theOdds = az keyvault secret show --vault-name $SourceKeyVaultName --name THE-ODDS-API-KEY --query value -o tsv
$apiBasket = az keyvault secret show --vault-name $SourceKeyVaultName --name API-BASKETBALL-KEY --query value -o tsv
if (-not $theOdds) { Write-Error "Missing THE-ODDS-API-KEY in source KV $SourceKeyVaultName"; exit 1 }

az keyvault secret set --vault-name $DestKeyVaultName --name THE-ODDS-API-KEY --value $theOdds --output none
if ($apiBasket) { az keyvault secret set --vault-name $DestKeyVaultName --name API-BASKETBALL-KEY --value $apiBasket --output none }
Write-Host "  ✓ Secrets cloned" -ForegroundColor Green

Write-Host "[5/6] Provision/Update NBA Container App in $DestResourceGroup" -ForegroundColor Yellow
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$provisionScript = Join-Path $scriptDir 'provision.ps1'

& $provisionScript -ResourceGroup $DestResourceGroup -SharedResourceGroup $DestResourceGroup -EnvironmentName $ContainerAppEnvName -AppName $ContainerAppName -AcrName $DestAcrName -Image "${destAcrServer}/${ImageName}:${Tag}" -KeyVaultName $DestKeyVaultName -Location $Location

if ($LASTEXITCODE -ne 0) { Write-Error "Provisioning NBA Container App failed"; exit 1 }

Write-Host "[6/6] Update Container App image and secrets linkage" -ForegroundColor Yellow
az containerapp update -n $ContainerAppName -g $DestResourceGroup --image "${destAcrServer}/${ImageName}:${Tag}" --output none

$fqdn = az containerapp show -n $ContainerAppName -g $DestResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host "  ✓ NBA API FQDN: https://$fqdn" -ForegroundColor Green
Write-Host "  Test: curl -s https://$fqdn/health" -ForegroundColor Yellow
