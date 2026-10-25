param location string
param name string
param tags object

module registry 'br/public:avm/res/container-registry/registry:0.13.1' = {
  name: 'registry'
  params: {
    acrAdminUserEnabled: false
    acrSku: 'Basic'
    enableTelemetry: false
    location: location
    name: name
    networkRuleBypassOptions: 'AzureServices'
    networkRuleSetDefaultAction: 'Allow'
    publicNetworkAccess: 'Enabled'
    retentionPolicyStatus: 'disabled'
    softDeletePolicyStatus: 'disabled'
    tags: tags
    zoneRedundancy: 'Disabled'
  }
}

output loginServer string = registry.outputs.loginServer
output resourceId string = registry.outputs.resourceId
