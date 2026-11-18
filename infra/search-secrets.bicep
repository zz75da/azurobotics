targetScope = 'resourceGroup'

// Stores the Azure AI Search keys in Key Vault. The Free tier has no Entra authentication, so
// keys are unavoidable; they are read from the service at deployment time and never pass through
// the repository. search-query-key is read (by reference) by the search Container App, through a
// role scoped to that one secret. search-admin-key is only for the indexing script, run by a person.

param searchName string
param keyVaultName string

resource search 'Microsoft.Search/searchServices@2025-05-01' existing = {
  name: searchName
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' existing = {
  name: keyVaultName
}

resource queryKey 'Microsoft.KeyVault/vaults/secrets@2026-02-01' = {
  parent: keyVault
  name: 'search-query-key'
  properties: {
    value: search.listQueryKeys().value[0].key
    contentType: 'Azure AI Search query key (read-only queries)'
  }
}

resource adminKey 'Microsoft.KeyVault/vaults/secrets@2026-02-01' = {
  parent: keyVault
  name: 'search-admin-key'
  properties: {
    value: search.listAdminKeys().primaryKey
    contentType: 'Azure AI Search admin key (indexing script only)'
  }
}

output queryKeySecretUri string = queryKey.properties.secretUri
