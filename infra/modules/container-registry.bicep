param location string
param name string
param tags object

module registry 'br/public:avm/res/container-registry/registry:0.13.1' = {
  name: 'registry'
  params: {
    acrAdminUserEnabled: false
    acrSku: 'Basic'
    // Accept ARM-scoped Entra tokens as well as registry-scoped ones. Azure ML compute clusters
    // pull training images with ARM-scoped tokens: with this policy disabled (the module default,
    // and Microsoft's recommended hardening) every pull failed with "could not authenticate with
    // the Docker registry", even with AcrPull granted. The admin user stays disabled.
    azureADAuthenticationAsArmPolicyStatus: 'enabled'
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
