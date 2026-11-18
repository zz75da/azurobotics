targetScope = 'resourceGroup'

// Azure AI Search for the maintenance-procedures demo (personal project, synthetic notes).
// Free tier first: 50 MB, 3 indexes, no SLA, one free service per subscription. Microsoft does not
// document a vector quota for Free, so if vector indexing fails the demo moves to Basic only for
// the duration of a demo (billed hourly) and is torn down afterwards (scripts/teardown_azureml_search.sh).
// Free has no Microsoft Entra / managed-identity authentication, so access uses API keys, which are
// stored in Key Vault in a later step and never in the repository.

param location string = resourceGroup().location
param searchName string

@allowed(['free', 'basic'])
param sku string = 'free'

param tags object = {}

resource search 'Microsoft.Search/searchServices@2025-05-01' = {
  name: searchName
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'Default'
    publicNetworkAccess: 'enabled'
    // Keys only: the Free tier cannot use Entra ID authentication.
    authOptions: {
      apiKeyOnly: {}
    }
    semanticSearch: 'disabled'
  }
}

output searchName string = search.name
output endpoint string = 'https://${search.name}.search.windows.net'
