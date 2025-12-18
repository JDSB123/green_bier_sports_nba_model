param(
    [string]$FunctionAppName,
    [string]$ResourceGroup,
    [string]$Location = "eastus"
)

$ErrorActionPreference = "Stop"

# Create Resource Group
Write-Host "Creating Resource Group '$ResourceGroup'..."
az group create --name $ResourceGroup --location $Location

# Create Storage Account
# Storage names must be lowercase alphanumeric, max 24 chars
$storageName = ($FunctionAppName -replace "-","").ToLower()
if ($storageName.Length -gt 20) {
    $storageName = $storageName.Substring(0, 20)
}
$storageName = "${storageName}st"

Write-Host "Creating Storage Account '$storageName'..."
az storage account create --name $storageName --resource-group $ResourceGroup --location $Location --sku Standard_LRS

# Create Function App (Consumption)
Write-Host "Creating Function App '$FunctionAppName'..."
az functionapp create `
    --name $FunctionAppName `
    --resource-group $ResourceGroup `
    --storage-account $storageName `
    --consumption-plan-location $Location `
    --runtime python `
    --runtime-version 3.11 `
    --functions-version 4 `
    --os-type Linux

Write-Host "✅ Function App '$FunctionAppName' created successfully."
