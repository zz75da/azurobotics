param location string
param name string
param tags object
param tenantId string

resource vault 'Microsoft.KeyVault/vaults@2026-05-15' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenantId
    enableRbacAuthorization: true
    softDeleteRetentionInDays: 7
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Allow'
    }
    publicNetworkAccess: 'Enabled'
  }
}

resource entraClientSecret 'Microsoft.KeyVault/vaults/secrets@2026-02-01' existing = {
  parent: vault
  name: 'entra-client-secret'
}

output resourceId string = vault.id
output vaultUri string = vault.properties.vaultUri
output entraClientSecretUri string = '${vault.properties.vaultUri}secrets/${entraClientSecret.name}'
