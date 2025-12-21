// Green Bier Sports - NBA Model Infrastructure
// Deploys NBA-specific resources (Container App, Function, Storage)
//
// SINGLE SOURCE OF TRUTH - All NBA resources in greenbier-enterprise-rg
// Container App: nba-picks-api
// ACR: greenbieracr
//
// Usage:
//   az deployment group create -g greenbier-enterprise-rg -f infra/nba/main.bicep

targetScope = 'resourceGroup'

@description('Azure region')
param location string = resourceGroup().location

@description('Environment')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Shared resource group name - same as NBA for simplicity')
param sharedResourceGroup string = 'greenbier-enterprise-rg'

@description('Container image tag')
param imageTag string = 'latest'

// =============================================================================
// API KEYS - Required for the NBA Picks API to function
// =============================================================================
@description('The Odds API Key (required)')
@secure()
param theOddsApiKey string

@description('API-Basketball Key (required)')
@secure()
param apiBasketballKey string

@description('Teams Webhook URL (optional)')
@secure()
param teamsWebhookUrl string = ''

// Reference shared resources
resource sharedRg 'Microsoft.Resources/resourceGroups@2021-04-01' existing = {
  name: sharedResourceGroup
  scope: subscription()
}

// Get Container Apps Environment - ACTUAL: greenbier-nba-env
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: 'greenbier-nba-env'
  scope: resourceGroup(sharedResourceGroup)
}

// Get shared Container Registry - greenbieracr in greenbier-enterprise-rg
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: 'greenbieracr'
  scope: resourceGroup(sharedResourceGroup)
}

// Get shared Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: 'gbs-keyvault-${uniqueString(sharedRg.id)}'
  scope: resourceGroup(sharedResourceGroup)
}

// Get shared App Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: 'gbs-insights-${environment}'
  scope: resourceGroup(sharedResourceGroup)
}

// Naming
var sport = 'nba'
var tags = {
  enterprise: 'green-bier-sports-ventures'
  sport: sport
  environment: environment
  managedBy: 'bicep'
}

// ============================================================================
// Storage Account (NBA predictions archive)
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
  properties: {
    publicAccess: 'None'
  }
}

resource predictionsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'predictions'
  properties: {
    publicAccess: 'None'
  }
}

resource resultsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'results'
  properties: {
    publicAccess: 'None'
  }
}

// ============================================================================
// Container App - NBA Picks API (ACTUAL NAME: nba-picks-api)
// ============================================================================
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'nba-picks-api'
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
        {
          name: 'the-odds-api-key'
          value: theOddsApiKey
        }
        {
          name: 'api-basketball-key'
          value: apiBasketballKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'gbs-nba-api'
          image: '${acr.properties.loginServer}/nba-model:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            // ================================================================
            // REQUIRED API KEYS - Referenced from secrets
            // ================================================================
            {
              name: 'THE_ODDS_API_KEY'
              secretRef: 'the-odds-api-key'
            }
            {
              name: 'API_BASKETBALL_KEY'
              secretRef: 'api-basketball-key'
            }
            // ================================================================
            // NBA v6.4 Configuration
            // ================================================================
            {
              name: 'GBS_SPORT'
              value: 'nba'
            }
            {
              name: 'NBA_MODEL_VERSION'
              value: '6.5-STRICT'
            }
            {
              name: 'NBA_MARKETS'
              value: 'q1_spread,q1_total,q1_moneyline,1h_spread,1h_total,1h_moneyline,fg_spread,fg_total,fg_moneyline'
            }
            {
              name: 'NBA_PERIODS'
              value: 'first_quarter,first_half,full_game'
            }
            // ================================================================
            // Azure Integration
            // ================================================================
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
              httpGet: {
                path: '/health'
                port: 8080
              }
              initialDelaySeconds: 10
              periodSeconds: 30
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8080
              }
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
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
}

// ============================================================================
// Function App - NBA Picks Trigger
// ============================================================================
resource functionStorageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'gbs${sport}func${uniqueString(resourceGroup().id)}'
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
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
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  properties: {
    reserved: true
  }
}

resource functionApp 'Microsoft.Web/sites@2022-09-01' = {
  name: 'gbs-${sport}-trigger-${environment}'
  location: location
  tags: tags
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
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
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        {
          name: 'GBS_SPORT'
          value: 'nba'
        }
        {
          name: 'NBA_API_URL'
          value: 'https://${containerApp.properties.configuration.ingress.fqdn}'
        }
        // TEAMS_WEBHOOK_URL should be set manually or via Key Vault reference
      ]
    }
    httpsOnly: true
  }
}

// ============================================================================
// Key Vault Access for Container App
// ============================================================================
resource keyVaultAccessPolicy 'Microsoft.KeyVault/vaults/accessPolicies@2023-07-01' = {
  name: 'add'
  parent: keyVault
  properties: {
    accessPolicies: [
      {
        tenantId: subscription().tenantId
        objectId: containerApp.identity.principalId
        permissions: {
          secrets: ['get', 'list']
        }
      }
      {
        tenantId: subscription().tenantId
        objectId: functionApp.identity.principalId
        permissions: {
          secrets: ['get', 'list']
        }
      }
    ]
  }
}

// ============================================================================
// Outputs
// ============================================================================
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output functionAppUrl string = 'https://${functionApp.properties.defaultHostName}'
output functionAppName string = functionApp.name
output storageAccountName string = storageAccount.name
