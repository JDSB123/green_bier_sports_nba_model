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
        {
          name: 'database-url'
          value: databaseUrl
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
            {
              name: 'DATABASE_URL'
              secretRef: 'database-url'
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
output postgresServerFqdn string = postgresServer.properties.fullyQualifiedDomainName
output databaseName string = 'gbs_picks'
