param(
    [Parameter(Mandatory = $true)]
    [string]$ResourceGroupName,

    [string]$OutputDir = "$(Split-Path -Parent $PSScriptRoot)\infra\baseline"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is required. Install from https://learn.microsoft.com/cli/azure/install-azure-cli"
}

$requiredTags = @(
    'enterprise'
    'app'
    'environment'
    'owner'
    'cost_center'
    'compliance'
    'version'
    'managedBy'
)

if (-not (Test-Path -Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$reportPath = Join-Path $OutputDir "$ResourceGroupName-compliance-$timestamp.csv"

Write-Host "Scanning RG '$ResourceGroupName' for tag compliance..."
$resources = az resource list -g $ResourceGroupName --output json | ConvertFrom-Json
$rows = foreach ($r in $resources) {
    $missing = @()
    foreach ($tag in $requiredTags) {
        if (-not $r.tags.ContainsKey($tag)) {
            $missing += $tag
        }
    }

    [PSCustomObject]@{
        name                = $r.name
        type                = $r.type
        location            = $r.location
        missingRequiredTags = if ($missing.Count -gt 0) { $missing -join ';' } else { '' }
    }
}

$rows | Export-Csv -Path $reportPath -NoTypeInformation -Encoding UTF8
Write-Host "Compliance report written to $reportPath"
Write-Host "Tip: use this with the baseline export before cleanup/remediation."
