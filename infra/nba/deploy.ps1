Param(
    [string]$Subscription = "",
    [string]$ResourceGroup = "NBAGBSVMODEL",
    [string]$ContainerAppName = "nba-picks-api",
    [string]$AcrName = "greenbieracr",
    [string]$ImageName = "nba-model",
    [string]$Tag = "v6.4"
)

Write-Host "Starting NBA deployment to Azure Container Apps..."

if ($Subscription -and $Subscription.Trim() -ne "") {
    az account set --subscription $Subscription
    if ($LASTEXITCODE -ne 0) { Write-Error "Failed to set Azure subscription: $Subscription"; exit 1 }
} else {
    Write-Host "Using current Azure subscription:"
    az account show --query name -o tsv
}

# Login to Azure (assumes az login already performed if session is active)
az acr login -n $AcrName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to login to ACR: $AcrName"; exit 1
}

$fullImage = "$AcrName.azurecr.io/$ImageName:$Tag"
Write-Host "Building Docker image: $fullImage"

# Determine repo root (deploy.ps1 is expected in infra/nba/, Dockerfile at repo root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
Write-Host "Repository root: $RepoRoot"

Push-Location $RepoRoot
try {
    docker build -t $fullImage .
    if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed"; exit 1 }
} finally {
    Pop-Location
}

Write-Host "Pushing image: $fullImage"
docker push $fullImage
if ($LASTEXITCODE -ne 0) { Write-Error "Docker push failed"; exit 1 }

Write-Host "Verifying Container App exists: $ContainerAppName in RG: $ResourceGroup"
$null = az containerapp show -n $ContainerAppName -g $ResourceGroup --only-show-errors
if ($LASTEXITCODE -ne 0) {
    Write-Error "Container App '$ContainerAppName' not found in resource group '$ResourceGroup'. Please create it or verify the subscription/resource group."; exit 2
}

Write-Host "Updating Container App: $ContainerAppName in RG: $ResourceGroup"
az containerapp update -n $ContainerAppName -g $ResourceGroup --image $fullImage
if ($LASTEXITCODE -ne 0) { Write-Error "Container App update failed"; exit 1 }

Write-Host "Deployment complete. Optionally verify health endpoint."