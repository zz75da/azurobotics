param location string
param name string
param tags object
param workspaceCustomerId string
@secure()
param workspaceSharedKey string

resource environment 'Microsoft.App/managedEnvironments@2026-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspaceCustomerId
        sharedKey: workspaceSharedKey
      }
    }
  }
}

output resourceId string = environment.id
output defaultDomain string = environment.properties.defaultDomain
