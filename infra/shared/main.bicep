// Green Bier Sports Ventures - Shared Infrastructure
// Deploy ONCE - shared across all sports (NBA, NCAAM, NFL, MLB, etc.)
//
// SINGLE SOURCE OF TRUTH:
//   Resource Group: greenbier-enterprise-rg
//   ACR:            greenbieracr
//
// Usage:
//   az deployment group create -g greenbier-enterprise-rg -f infra/shared/main.bicep

targetScope = 'resourceGroup'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Environment (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

// Naming
var prefix = 'gbs'
var tags = {
  enterprise: 'green-bier-sports-ventures'
  environment: environment
  managedBy: 'bicep'
}

// ============================================================================
// Container Registry (shared across all sports) - ACTUAL: greenbieracr
// ============================================================================
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'greenbieracr'
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
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${prefix}-keyvault-${uniqueString(resourceGroup().id)}'
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
    enablePurgeProtection: false // Set to true for production
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
// Container Apps Environment - ACTUAL: greenbier-nba-env
// ============================================================================
resource containerAppEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: 'greenbier-nba-env'
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
