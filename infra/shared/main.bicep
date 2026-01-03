// Green Bier Sports Ventures - Shared Infrastructure
// Deploy ONCE - shared across all sports (NBA, NCAAM, NFL, MLB, etc.)
//
// Usage:
//   az deployment group create -g <shared-rg> -f infra/shared/main.bicep

targetScope = 'resourceGroup'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Environment (dev, staging, prod)')
@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string = 'prod'

@description('Semantic version for shared stack (tagging only)')
param versionTag string = 'shared-1.0.0'

@description('Application tag identifier')
param appTag string = 'shared-platform'

@description('Owner tag value')
param ownerTag string = 'platform-eng'

@description('Cost center tag value')
param costCenterTag string = 'platform-shared'

@description('Compliance tag value')
param complianceTag string = 'internal'

@description('Additional tags to merge onto all resources')
param extraTags object = {}

@description('Container Registry name (override to clone/shared per RG)')
param containerRegistryName string = 'nbagbsacr'

@description('Key Vault name override (defaults to unique computed name when empty)')
param keyVaultName string = ''

@description('Container Apps Environment name')
param containerAppEnvName string = 'nba-gbsv-model-env'

// Naming
var prefix = 'gbs'
var requiredTags = {
  enterprise: 'green-bier-sports-ventures'
  environment: environment
  app: appTag
  owner: ownerTag
  cost_center: costCenterTag
  compliance: complianceTag
  version: versionTag
  managedBy: 'bicep'
}
var tags = union(requiredTags, extraTags)

// ============================================================================
// Container Registry (shared across all sports) - ACTUAL: nbagbsacr
// ============================================================================
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// ============================================================================
// Key Vault (centralized secrets for all sports)
// ============================================================================
var resolvedKeyVaultName = empty(keyVaultName) ? '${prefix}-keyvault-${uniqueString(resourceGroup().id)}' : keyVaultName

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: resolvedKeyVaultName
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enablePurgeProtection: true
  }
}

// ============================================================================
// Log Analytics Workspace (centralized logging)
// ============================================================================
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${prefix}-logs-${environment}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// ============================================================================
// Application Insights (centralized telemetry)
// ============================================================================
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-insights-${environment}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ============================================================================
// Container Apps Environment - ACTUAL: nba-gbsv-model-env
// ============================================================================
resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ============================================================================
// Outputs (used by sport-specific deployments)
// ============================================================================
output containerRegistryName string = containerRegistry.name
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
output logAnalyticsWorkspaceId string = logAnalytics.id
output logAnalyticsCustomerId string = logAnalytics.properties.customerId
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
output containerAppEnvironmentId string = containerAppEnvironment.id
output containerAppEnvironmentName string = containerAppEnvironment.name
