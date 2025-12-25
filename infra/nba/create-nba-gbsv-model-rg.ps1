# NBA Production Deployment Script
# Creates the new nba-gbsv-model-rg resource group and provisions all resources
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - THE_ODDS_API_KEY and API_BASKETBALL_KEY secrets available
#
# Usage:
#   .\create-nba-gbsv-model-rg.ps1
#   .\create-nba-gbsv-model-rg.ps1 -Location "eastus" -Tag "v6.10"

Param(
    [string]$Location = "eastus",
    [string]$ResourceGroup = "nba-gbsv-model-rg",
    [string]$AcrName = "nbagbsacr",
    [string]$KeyVaultName = "nbagbs-keyvault",
    [string]$ContainerAppEnvName = "nba-gbsv-model-env",
    [string]$ContainerAppName = "nba-gbsv-api",
    [string]$Tag = "v6.10"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  NBA GBSV Model - Production Resource Group Setup" -ForegroundColor Green
Write-Host "  Target: $ResourceGroup" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Check Azure CLI login
Write-Host "[1/7] Checking Azure CLI..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "  -> Not logged in. Running 'az login'..." -ForegroundColor Yellow
    az login
    $account = az account show | ConvertFrom-Json
}
Write-Host "  Logged in: $($account.user.name)" -ForegroundColor Green
Write-Host "  Subscription: $($account.name)" -ForegroundColor Green
Write-Host ""

# Create Resource Group
Write-Host "[2/7] Creating Resource Group: $ResourceGroup..." -ForegroundColor Yellow
az group create --name $ResourceGroup --location $Location --output none
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create resource group"; exit 1 }
Write-Host "  Resource Group created in $Location" -ForegroundColor Green
Write-Host ""

# Create Container Registry
Write-Host "[3/7] Creating Container Registry: $AcrName..." -ForegroundColor Yellow
$acrExists = az acr show --name $AcrName --resource-group $ResourceGroup 2>$null
if (-not $acrExists) {
    az acr create --name $AcrName --resource-group $ResourceGroup --sku Basic --admin-enabled true --output none
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create ACR"; exit 1 }
    Write-Host "  ACR created: $AcrName.azurecr.io" -ForegroundColor Green
} else {
    Write-Host "  ACR already exists: $AcrName.azurecr.io" -ForegroundColor Cyan
}
Write-Host ""

# Create Key Vault
Write-Host "[4/7] Creating Key Vault: $KeyVaultName..." -ForegroundColor Yellow
$kvExists = az keyvault show --name $KeyVaultName --resource-group $ResourceGroup 2>$null
if (-not $kvExists) {
    az keyvault create --name $KeyVaultName --resource-group $ResourceGroup --location $Location --output none
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create Key Vault"; exit 1 }
    Write-Host "  Key Vault created" -ForegroundColor Green
} else {
    Write-Host "  Key Vault already exists" -ForegroundColor Cyan
}

# Check for secrets
Write-Host "  Checking secrets..." -ForegroundColor Yellow
$oddsKey = $env:THE_ODDS_API_KEY
$basketKey = $env:API_BASKETBALL_KEY

if (-not $oddsKey -or -not $basketKey) {
    Write-Host "  API keys not in environment. Checking secrets folder..." -ForegroundColor Yellow
    $secretsDir = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..\secrets"
    if (Test-Path "$secretsDir\THE_ODDS_API_KEY") {
        $oddsKey = Get-Content "$secretsDir\THE_ODDS_API_KEY" -Raw
    }
    if (Test-Path "$secretsDir\API_BASKETBALL_KEY") {
        $basketKey = Get-Content "$secretsDir\API_BASKETBALL_KEY" -Raw
    }
}

if ($oddsKey -and $basketKey) {
    Write-Host "  Setting secrets in Key Vault..." -ForegroundColor Yellow
    az keyvault secret set --vault-name $KeyVaultName --name "THE-ODDS-API-KEY" --value $oddsKey.Trim() --output none
    az keyvault secret set --vault-name $KeyVaultName --name "API-BASKETBALL-KEY" --value $basketKey.Trim() --output none
    Write-Host "  Secrets configured" -ForegroundColor Green
} else {
    Write-Host "  WARNING: API keys not found. Set manually:" -ForegroundColor Yellow
    Write-Host "    az keyvault secret set --vault-name $KeyVaultName --name THE-ODDS-API-KEY --value '<key>'" -ForegroundColor White
    Write-Host "    az keyvault secret set --vault-name $KeyVaultName --name API-BASKETBALL-KEY --value '<key>'" -ForegroundColor White
}
Write-Host ""

# Create Container Apps Environment
Write-Host "[5/7] Creating Container Apps Environment: $ContainerAppEnvName..." -ForegroundColor Yellow
$envExists = az containerapp env show --name $ContainerAppEnvName --resource-group $ResourceGroup 2>$null
if (-not $envExists) {
    az containerapp env create --name $ContainerAppEnvName --resource-group $ResourceGroup --location $Location --output none
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create Container Apps Environment"; exit 1 }
    Write-Host "  Environment created" -ForegroundColor Green
} else {
    Write-Host "  Environment already exists" -ForegroundColor Cyan
}
Write-Host ""

# Build and push Docker image
Write-Host "[6/7] Building and pushing Docker image..." -ForegroundColor Yellow
$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..")
$fullImage = "$AcrName.azurecr.io/nba-gbsv-api:$Tag"

az acr login --name $AcrName
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to login to ACR"; exit 1 }

Push-Location $repoRoot
try {
    Write-Host "  Building image: $fullImage" -ForegroundColor Cyan
    docker build -f Dockerfile.combined -t $fullImage .
    if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed"; exit 1 }

    Write-Host "  Pushing image..." -ForegroundColor Cyan
    docker push $fullImage
    if ($LASTEXITCODE -ne 0) { Write-Error "Docker push failed"; exit 1 }
    Write-Host "  Image pushed: $fullImage" -ForegroundColor Green
} finally {
    Pop-Location
}
Write-Host ""

# Create/Update Container App
Write-Host "[7/7] Creating Container App: $ContainerAppName..." -ForegroundColor Yellow
$appExists = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup 2>$null
if (-not $appExists) {
    # Get secrets from Key Vault
    $oddsApiKey = az keyvault secret show --vault-name $KeyVaultName --name THE-ODDS-API-KEY --query value -o tsv
    $apiBasketballKey = az keyvault secret show --vault-name $KeyVaultName --name API-BASKETBALL-KEY --query value -o tsv

    az containerapp create `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --environment $ContainerAppEnvName `
        --image $fullImage `
        --target-port 8080 `
        --ingress external `
        --min-replicas 1 `
        --max-replicas 10 `
        --cpu 0.5 `
        --memory 1Gi `
        --secrets the-odds-api-key=$oddsApiKey api-basketball-key=$apiBasketballKey `
        --env-vars THE_ODDS_API_KEY=secretref:the-odds-api-key API_BASKETBALL_KEY=secretref:api-basketball-key `
        --output none

    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create Container App"; exit 1 }
    Write-Host "  Container App created" -ForegroundColor Green
} else {
    Write-Host "  Container App exists, updating image..." -ForegroundColor Cyan
    az containerapp update --name $ContainerAppName --resource-group $ResourceGroup --image $fullImage --output none
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to update Container App"; exit 1 }
    Write-Host "  Container App updated" -ForegroundColor Green
}
Write-Host ""

# Get FQDN and verify
$fqdn = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Resource Group:     $ResourceGroup" -ForegroundColor Cyan
Write-Host "  Container Registry: $AcrName.azurecr.io" -ForegroundColor Cyan
Write-Host "  Key Vault:          $KeyVaultName" -ForegroundColor Cyan
Write-Host "  Container App:      $ContainerAppName" -ForegroundColor Cyan
Write-Host "  API URL:            https://$fqdn" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Verify with:" -ForegroundColor Yellow
Write-Host "    curl https://$fqdn/health" -ForegroundColor White
Write-Host ""
