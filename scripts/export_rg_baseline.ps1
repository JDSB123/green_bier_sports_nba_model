param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroupName,

    [string]$OutputDir = "$(Split-Path -Parent $PSScriptRoot)\infra\baseline",

    [switch]$IncludeRoleAssignments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is required. Install from https://learn.microsoft.com/cli/azure/install-azure-cli"
}

if (-not (Test-Path -Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$baselinePath = Join-Path $OutputDir "$ResourceGroupName-baseline-$timestamp.json"
$tagReportPath = Join-Path $OutputDir "$ResourceGroupName-tags-$timestamp.csv"
$rolePath = Join-Path $OutputDir "$ResourceGroupName-role-assignments-$timestamp.json"

Write-Host "Exporting resource inventory for RG '$ResourceGroupName'..."
$resourcesJson = az resource list -g $ResourceGroupName --output json
Set-Content -Path $baselinePath -Value $resourcesJson -Encoding UTF8

$resources = $resourcesJson | ConvertFrom-Json
$tagRows = $resources | ForEach-Object {
    [PSCustomObject]@{
        name      = $_.name
        type      = $_.type
        location  = $_.location
        tags      = ($_ | Select-Object -ExpandProperty tags | ConvertTo-Json -Compress)
        managedBy = $_.managedBy
    }
}

Write-Host "Writing tag report to $tagReportPath"
$tagRows | Export-Csv -Path $tagReportPath -NoTypeInformation -Encoding UTF8

if ($IncludeRoleAssignments) {
    Write-Host "Exporting role assignments..."
    $roleAssignments = az role assignment list --resource-group $ResourceGroupName --output json
    Set-Content -Path $rolePath -Value $roleAssignments -Encoding UTF8
}

Write-Host "Baseline export complete."
Write-Host "  Inventory : $baselinePath"
Write-Host "  Tags      : $tagReportPath"
if ($IncludeRoleAssignments) {
    Write-Host "  Roles     : $rolePath"
}
