// Green Bier Sports - NBA Model Infrastructure
// Deploys NBA-specific resources (Container App, Function, Storage)
//
// SINGLE SOURCE OF TRUTH - All NBA resources in nba-gbsv-model-rg
// Container App: nba-gbsv-api
// ACR: nbagbsacr (in nba-gbsv-model-rg, NOT shared)
// Key Vault: nbagbs-keyvault
//
// Usage:
//   az deployment group create -g nba-gbsv-model-rg -f infra/nba/main.bicep

targetScope = 'resourceGroup'

@description('Azure region')
param location string = resourceGroup().location

@description('Environment')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Container image tag')
param imageTag string = 'NBA_v33.0.8.0'

@description('Database URL (optional)')
@secure()
param databaseUrl string = ''

@description('Application Insights connection string (optional)')
@secure()
param appInsightsConnectionString string = ''

// =============================================================================
// API KEYS - Required for the NBA Picks API to function
// =============================================================================
@description('The Odds API Key (required)')
@secure()
param theOddsApiKey string

@description('API-Basketball Key (required)')
@secure()
param apiBasketballKey string

@description('Website domain for CORS (e.g., greenbiersportventures.com)')
param websiteDomain string = 'greenbiersportventures.com'

// Get Container Apps Environment - nbagbsvmodel-env (NOT greenbier-nba-env)
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: 'nbagbsvmodel-env'
}

// Get Container Registry - nbagbsacr (in same resource group)
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: 'nbagbsacr'
}

// Naming
var sport = 'nba'
var tags = {
  enterprise: 'green-bier-sports-ventures'
  sport: sport
  environment: environment
  managedBy: 'bicep'
}

// Secrets list (optional entries appended)
var apiSecrets = concat(
  [
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
  ],
  databaseUrl == '' ? [] : [
    {
      name: 'database-url'
      value: databaseUrl
    }
  ],
  appInsightsConnectionString == '' ? [] : [
    {
      name: 'app-insights-connection-string'
      value: appInsightsConnectionString
    }
  ]
)

// Environment variables (optional entries appended)
var appEnvVars = concat(
  [
    // Required API keys via secrets
    {
      name: 'THE_ODDS_API_KEY'
      secretRef: 'the-odds-api-key'
    }
    {
      name: 'API_BASKETBALL_KEY'
      secretRef: 'api-basketball-key'
    }
    // NBA v33.0.8.0 configuration
    {
      name: 'GBS_SPORT'
      value: 'nba'
    }
    {
      name: 'NBA_MODEL_VERSION'
      value: 'NBA_v33.0.8.0'
    }
    {
      name: 'NBA_MARKETS'
      value: '1h_spread,1h_total,fg_spread,fg_total'
    }
    {
      name: 'NBA_PERIODS'
      value: 'first_half,full_game'
    }
    // Azure storage (inline connection string)
    {
      name: 'AZURE_STORAGE_CONNECTION_STRING'
      value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
    }
  ],
  databaseUrl == '' ? [] : [
    {
      name: 'DATABASE_URL'
      secretRef: 'database-url'
    }
  ],
  appInsightsConnectionString == '' ? [] : [
    {
      name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
      secretRef: 'app-insights-connection-string'
    }
  ]
)

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
// Container App - NBA Picks API (ACTUAL NAME: nba-gbsv-api)
// ============================================================================
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'nba-gbsv-api'
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
            'https://${websiteDomain}'
            'https://www.${websiteDomain}'
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
      secrets: apiSecrets
    }
    template: {
      containers: [
        {
          name: 'gbs-nba-api'
          image: '${acr.properties.loginServer}/nba-gbsv-api:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: appEnvVars
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
        minReplicas: 1
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
// Outputs
// ============================================================================
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output storageAccountName string = storageAccount.name
