param location string
param name string
param tags object

module workspace 'br/public:avm/res/operational-insights/workspace:0.16.1' = {
  name: 'workspace'
  params: {
    dailyQuotaGb: '0.16'
    dataRetention: 30
    enableTelemetry: false
    forceCmkForQuery: false
    location: location
    name: name
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
    skuName: 'PerGB2018'
    tags: tags
  }
}

output customerId string = workspace.outputs.logAnalyticsWorkspaceId
output resourceId string = workspace.outputs.resourceId
