targetScope = 'subscription'

@minLength(1)
@maxLength(64)
param environmentName string

@minLength(1)
param location string

param sessionId string
param deployedBy string
param createdAt string
param deployerObjectId string
param tenantId string
param resourceGroupName string
param containerAppsEnvironmentName string
param containerAppName string
param registryName string
param logAnalyticsName string
param keyVaultName string
param entraClientId string = ''
param containerImage string = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param enableIngress bool = false
param mlflowStorageName string = ''

var tags = {
  'app-onboard-skill': 'true'
  'app-onboard-session-id': sessionId
  'created-at': createdAt
  environment: environmentName
  'deployed-by': deployedBy
}

var appHostName = '${containerAppName}.${containerAppsEnvironment.outputs.defaultDomain}'
var isPlaceholder = containerImage == 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
var enablePublicIngress = enableIngress && !isPlaceholder

resource rg 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module registry './modules/container-registry.bicep' = {
  name: 'containerRegistry'
  scope: rg
  params: {
    location: location
    name: registryName
    tags: tags
  }
}

module logAnalytics './modules/log-analytics.bicep' = {
  name: 'logAnalytics'
  scope: rg
  params: {
    location: location
    name: logAnalyticsName
    tags: tags
  }
}

module keyVault './modules/key-vault.bicep' = {
  name: 'keyVault'
  scope: rg
  params: {
    location: location
    name: keyVaultName
    tags: tags
    tenantId: tenantId
  }
}

module containerAppsEnvironment './modules/container-app-environment.bicep' = {
  name: 'containerAppsEnvironment'
  scope: rg
  params: {
    location: location
    name: containerAppsEnvironmentName
    tags: tags
    workspaceResourceId: logAnalytics.outputs.resourceId
  }
}

module containerApp './modules/container-app.bicep' = {
  name: 'containerApp'
  scope: rg
  params: {
    appHostName: appHostName
    containerImage: containerImage
    containerAppName: containerAppName
    containerAppsEnvironmentResourceId: containerAppsEnvironment.outputs.resourceId
    entraClientId: entraClientId
    enablePublicIngress: enablePublicIngress
    isPlaceholder: isPlaceholder
    keyVaultSecretUri: keyVault.outputs.entraClientSecretUri
    location: location
    mlflowStorageName: mlflowStorageName
    registryLoginServer: registry.outputs.loginServer
    tags: tags
    tenantId: tenantId
  }
}

module roleAssignments './modules/role-assignments.bicep' = {
  name: 'roleAssignments'
  scope: rg
  params: {
    appPrincipalId: containerApp.outputs.systemAssignedMIPrincipalId!
    deployerObjectId: deployerObjectId
    keyVaultName: keyVaultName
    registryName: registryName
  }
}

output resourceGroupName string = rg.name
output registryLoginServer string = registry.outputs.loginServer
output containerAppName string = containerApp.outputs.name
output containerAppFqdn string = containerApp.outputs.fqdn
output plannedContainerAppFqdn string = appHostName
output keyVaultUri string = keyVault.outputs.vaultUri
