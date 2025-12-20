// Green Bier Sports - NCAAM Model Infrastructure
// Copy this to your NCAAM workspace and deploy
//
// Prerequisites: Deploy shared infrastructure first (infra/shared/main.bicep)
//
// Usage:
//   az group create -n gbs-ncaam-rg -l eastus
//   az deployment group create -g gbs-ncaam-rg -f infra/ncaam/main.bicep \
//     --parameters sharedResourceGroup=gbs-shared-rg

targetScope = 'resourceGroup'

@description('Azure region')
param location string = resourceGroup().location

@description('Environment')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Shared resource group name')
param sharedResourceGroup string = 'gbs-shared-rg'

@description('Container image tag')
param imageTag string = 'latest'

// Reference shared resources
resource sharedRg 'Microsoft.Resources/resourceGroups@2021-04-01' existing = {
  name: sharedResourceGroup
  scope: subscription()
}

resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: 'gbs-apps-env-${environment}'
  scope: resourceGroup(sharedResourceGroup)
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: 'gbssportsacr'
  scope: resourceGroup(sharedResourceGroup)
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: 'gbs-insights-${environment}'
  scope: resourceGroup(sharedResourceGroup)
}

// Naming
var sport = 'ncaam'
var tags = {
  enterprise: 'green-bier-sports-ventures'
  sport: sport
  environment: environment
  managedBy: 'bicep'
}

// ============================================================================
// Storage Account (NCAAM predictions archive)
// ============================================================================
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'gbs${sport}data${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: 'Hot'
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'models'
  properties: { publicAccess: 'None' }
}

resource predictionsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'predictions'
  properties: { publicAccess: 'None' }
}

resource resultsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'results'
  properties: { publicAccess: 'None' }
}

// ============================================================================
// Container App - NCAAM Picks API
// ============================================================================
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'gbs-${sport}-api'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
        corsPolicy: {
          allowedOrigins: [
            'http://localhost:3000'
            'https://*.azurewebsites.net'
            'https://*.greenbier.com'
          ]
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'gbs-ncaam-api'
          image: '${acr.properties.loginServer}/ncaam-model:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'GBS_SPORT'
              value: 'ncaam'
            }
            {
              name: 'NCAAM_MODEL_VERSION'
              value: '1.0'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: appInsights.properties.ConnectionString
            }
            {
              name: 'AZURE_STORAGE_CONNECTION_STRING'
              value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
            }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8080 }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8080 }
              initialDelaySeconds: 5
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
        rules: [
          {
            name: 'http-rule'
            http: { metadata: { concurrentRequests: '50' } }
          }
        ]
      }
    }
  }
}

// ============================================================================
// Function App - NCAAM Picks Trigger
// ============================================================================
resource functionStorageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'gbs${sport}func${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: 'gbs-${sport}-plan-${environment}'
  location: location
  tags: tags
  sku: { name: 'Y1', tier: 'Dynamic' }
  properties: { reserved: true }
}

resource functionApp 'Microsoft.Web/sites@2022-09-01' = {
  name: 'gbs-${sport}-trigger-${environment}'
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      pythonVersion: '3.11'
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${functionStorageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${functionStorageAccount.listKeys().keys[0].value}'
        }
        { name: 'FUNCTIONS_EXTENSION_VERSION', value: '~4' }
        { name: 'FUNCTIONS_WORKER_RUNTIME', value: 'python' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
        { name: 'GBS_SPORT', value: 'ncaam' }
        { name: 'NCAAM_API_URL', value: 'https://${containerApp.properties.configuration.ingress.fqdn}' }
      ]
    }
    httpsOnly: true
  }
}

// Outputs
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output functionAppUrl string = 'https://${functionApp.properties.defaultHostName}'
output storageAccountName string = storageAccount.name
