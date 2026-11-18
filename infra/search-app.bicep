targetScope = 'resourceGroup'

// Search API Container App (FastAPI + ONNX MiniLM) for the maintenance-procedures demo.
// Internal ingress only: reachable from inside the Container Apps environment, not from the
// internet, so no Entra sign-in is needed. Scales to zero.
// Its user-assigned identity has exactly two roles: AcrPull on the registry, and Key Vault Secrets
// User on the single secret search-query-key (not on the vault), read by reference.

param location string = resourceGroup().location
param appName string
param identityName string
param containerAppsEnvironmentName string
param registryName string
param keyVaultName string
param searchName string
param image string
param indexName string = 'maintenance-procedures'
param tags object = {}

var roles = {
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  keyVaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
}

resource environment 'Microsoft.App/managedEnvironments@2026-07-01' existing = {
  name: containerAppsEnvironmentName
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: registryName
}

resource keyVault 'Microsoft.KeyVault/vaults@2026-05-15' existing = {
  name: keyVaultName
}

resource queryKeySecret 'Microsoft.KeyVault/vaults/secrets@2026-02-01' existing = {
  parent: keyVault
  name: 'search-query-key'
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
  tags: tags
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identity.id, roles.acrPull)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.acrPull)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Scoped to one secret, so the app cannot read the vault's other secrets.
resource secretUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(queryKeySecret.id, identity.id, roles.keyVaultSecretsUser)
  scope: queryKeySecret
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.keyVaultSecretsUser)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource app 'Microsoft.App/containerApps@2026-01-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  dependsOn: [
    acrPull
    secretUser
  ]
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: false
        targetPort: 8000
        allowInsecure: false
        transport: 'auto'
      }
      registries: [
        {
          server: registry.properties.loginServer
          identity: identity.id
        }
      ]
      secrets: [
        {
          name: 'search-query-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/search-query-key'
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'search'
          image: image
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'SEARCH_ENDPOINT', value: 'https://${searchName}.search.windows.net' }
            { name: 'SEARCH_INDEX', value: indexName }
            { name: 'SEARCH_QUERY_KEY', secretRef: 'search-query-key' }
          ]
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 2
              periodSeconds: 3
              failureThreshold: 40
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

output internalUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output identityPrincipalId string = identity.properties.principalId
