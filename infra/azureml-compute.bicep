targetScope = 'resourceGroup'

// Training cluster for the Azure ML pipeline. With the registry's admin user disabled, the image
// is pulled with the cluster's managed identity. Serverless compute cannot use the workspace's
// system-assigned identity, and a user-assigned cluster identity with AcrPull still failed to pull,
// so the cluster uses a system-assigned identity: the case Microsoft recommends, where Azure ML
// grants AcrPull on the workspace registry itself. This template only adds blob data access on the
// workspace's dedicated storage. After deployment, the identity's roles are listed and anything
// beyond AcrPull on the registry and blob access on that storage is removed.
// The cluster scales between 0 and 1 node, so it costs nothing when idle.

param location string = resourceGroup().location
param workspaceName string
param computeName string = 'cpu-ds2'
param storageAccountName string
param vmSize string = 'Standard_DS2_v2'

@description('LowPriority needs low-priority quota; this subscription has none in the region.')
@allowed(['Dedicated', 'LowPriority'])
param vmPriority string = 'Dedicated'

param tags object = {}

var storageBlobDataContributor = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

resource workspace 'Microsoft.MachineLearningServices/workspaces@2025-09-01' existing = {
  name: workspaceName
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: storageAccountName
}

resource cluster 'Microsoft.MachineLearningServices/workspaces/computes@2025-09-01' = {
  parent: workspace
  name: computeName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
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

resource blob 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storage.id, cluster.id, storageBlobDataContributor)
  scope: storage
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataContributor)
    principalId: cluster.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

output computeName string = cluster.name
output clusterPrincipalId string = cluster.identity.principalId
