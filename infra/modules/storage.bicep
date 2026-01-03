// Storage module for NBA workloads
@description('Optional override for the storage account name')
param name string = ''

@description('Application identifier for naming (e.g., nba)')
param app string

@description('Environment (dev/staging/prod)')
param environment string

@description('Azure region')
param location string

@description('Tags to apply to all resources')
param tags object

@description('Blob container names to create')
param containers array = [
  'models'
  'predictions'
  'results'
]

@description('Storage SKU (default Standard_LRS)')
param skuName string = 'Standard_LRS'

@description('Storage account access tier')
param accessTier string = 'Hot'

var resolvedName = empty(name) ? 'gbs${app}data${uniqueString(resourceGroup().id)}' : toLower(name)

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: resolvedName
  location: location
  tags: tags
  sku: {
    name: skuName
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: accessTier
  }
}

resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource blobContainers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = [for containerName in containers: {
  parent: blobServices
  name: containerName
  properties: {
    publicAccess: 'None'
  }
}]

output storageAccountName string = storageAccount.name
output connectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${az.environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
