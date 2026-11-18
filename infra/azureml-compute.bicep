targetScope = 'resourceGroup'

// Training cluster for the Azure ML pipeline. Serverless compute cannot use the workspace's
// system-assigned identity, so it cannot pull images from a registry with the admin user
// disabled. This cluster pulls with its own user-assigned identity instead.
// The identity has exactly two roles: AcrPull on the registry and blob data access on the
// workspace's dedicated storage account. Nothing on the app's storage, Key Vault or the
// resource group. The cluster scales between 0 and 1 node, so it costs nothing when idle.

param location string = resourceGroup().location
param workspaceName string
param computeName string = 'cpu-ds2'
param identityName string
param registryName string
param storageAccountName string
param vmSize string = 'Standard_DS2_v2'

@allowed(['Dedicated', 'LowPriority'])
param vmPriority string = 'Dedicated'

param tags object = {}

var roles = {
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  storageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-09-01' existing = {
  name: workspaceName
}

resource registry 'Microsoft.ContainerRegistry/registries@2025-11-01' existing = {
  name: registryName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
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

resource blob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, identity.id, roles.storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.storageBlobDataContributor)
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource cluster 'Microsoft.MachineLearningServices/workspaces/computes@2025-09-01' = {
  parent: workspace
  name: computeName
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
    blob
  ]
  properties: {
    computeType: 'AmlCompute'
    description: 'Training cluster, scales to zero (personal project).'
    properties: {
      vmSize: vmSize
      vmPriority: vmPriority
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 1
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
}

output computeName string = cluster.name
output identityClientId string = identity.properties.clientId
