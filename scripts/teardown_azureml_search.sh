#!/usr/bin/env bash
# Delete the Azure ML and Azure AI Search resources added to the project, and nothing else.
# The app, MLflow, Grafana, Evidently, the app's storage account and Key Vault stay untouched,
# except for the two search-* secrets added to that Key Vault, which are deleted and purged.
#
#   scripts/teardown_azureml_search.sh [--yes] [--restore-registry-hardening]
#
# --restore-registry-hardening disables authentication-as-ARM on the registry again (it was
# enabled only so Azure ML clusters could pull images). Also set azureADAuthenticationAsArmPolicyStatus
# back to 'disabled' in infra/modules/container-registry.bicep if you do this.
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: keep /subscriptions/... paths intact

RG="${RG:-rg-marso-dev-99ff}"
WORKSPACE="${WORKSPACE:-mlw-robotics-dev-99ff}"
CLUSTER="${CLUSTER:-cpu-ds2}"
ML_STORAGE="${ML_STORAGE:-stmlwrobotics99ff}"
ML_VAULT="${ML_VAULT:-kv-mlw-robotics-99ff}"
APP_INSIGHTS="${APP_INSIGHTS:-appi-robotics-dev-99ff}"
SEARCH="${SEARCH:-srch-robotics-99ff}"
SEARCH_APP="${SEARCH_APP:-ca-search-dev-99ff}"
SEARCH_IDENTITY="${SEARCH_IDENTITY:-id-search-dev-99ff}"
APP_VAULT="${APP_VAULT:-kv-marso-dev-99ff}"
REGISTRY="${REGISTRY:-crmarsodev99ff}"

ASSUME_YES=false
RESTORE_HARDENING=false
for arg in "$@"; do
  case "$arg" in
    --yes) ASSUME_YES=true ;;
    --restore-registry-hardening) RESTORE_HARDENING=true ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cat <<EOF
This deletes, in resource group $RG:
  search app $SEARCH_APP and identity $SEARCH_IDENTITY, search service $SEARCH,
  secrets search-query-key and search-admin-key in $APP_VAULT (purged),
  Azure ML cluster $CLUSTER, workspace $WORKSPACE, storage $ML_STORAGE,
  Key Vault $ML_VAULT (purged), Application Insights $APP_INSIGHTS,
  role assignments held by those identities, and the robotics-search and azureml/* images in $REGISTRY.
EOF
if [ "$ASSUME_YES" != true ]; then
  read -r -p "Type the resource group name to confirm: " answer
  [ "$answer" = "$RG" ] || { echo "aborted"; exit 1; }
fi

principal_of() {  # principal id of a resource's identity, empty if it does not exist
  az resource show -g "$RG" -n "$1" --resource-type "$2" --query "${3:-identity.principalId}" -o tsv 2>/dev/null || true
}
remove_roles() {  # delete every role assignment held by a principal
  [ -n "$1" ] || return 0
  for id in $(az role assignment list --assignee-object-id "$1" --all --query "[].id" -o tsv); do
    az rest --method delete --url "https://management.azure.com${id}?api-version=2022-04-01" -o none
  done
}

echo "== search app"
SEARCH_PRINCIPAL=$(principal_of "$SEARCH_IDENTITY" Microsoft.ManagedIdentity/userAssignedIdentities properties.principalId)
az containerapp delete -g "$RG" -n "$SEARCH_APP" --yes -o none 2>/dev/null || true
remove_roles "$SEARCH_PRINCIPAL"
az identity delete -g "$RG" -n "$SEARCH_IDENTITY" -o none 2>/dev/null || true

echo "== search service and its secrets"
az search service delete -g "$RG" -n "$SEARCH" --yes -o none 2>/dev/null || true
for secret in search-query-key search-admin-key; do
  az keyvault secret delete --vault-name "$APP_VAULT" -n "$secret" -o none 2>/dev/null || true
  sleep 5
  az keyvault secret purge --vault-name "$APP_VAULT" -n "$secret" -o none 2>/dev/null || true
done

echo "== Azure ML"
WS_PRINCIPAL=$(principal_of "$WORKSPACE" Microsoft.MachineLearningServices/workspaces)
CLUSTER_PRINCIPAL=$(principal_of "$WORKSPACE/$CLUSTER" Microsoft.MachineLearningServices/workspaces/computes)
az resource delete -g "$RG" -n "$WORKSPACE/$CLUSTER" --resource-type Microsoft.MachineLearningServices/workspaces/computes -o none 2>/dev/null || true
remove_roles "$CLUSTER_PRINCIPAL"
az resource delete -g "$RG" -n "$WORKSPACE" --resource-type Microsoft.MachineLearningServices/workspaces -o none 2>/dev/null || true
remove_roles "$WS_PRINCIPAL"
az storage account delete -g "$RG" -n "$ML_STORAGE" --yes -o none 2>/dev/null || true
az keyvault delete -g "$RG" -n "$ML_VAULT" -o none 2>/dev/null || true
az keyvault purge -n "$ML_VAULT" -o none 2>/dev/null || true
az resource delete -g "$RG" -n "$APP_INSIGHTS" --resource-type Microsoft.Insights/components -o none 2>/dev/null || true

echo "== registry images"
for repo in $(az acr repository list -n "$REGISTRY" -o tsv | grep -E '^(robotics-search|azureml/)' || true); do
  az acr repository delete -n "$REGISTRY" --repository "$repo" --yes -o none
done
if [ "$RESTORE_HARDENING" = true ]; then
  az acr config authentication-as-arm update -r "$REGISTRY" --status disabled -o none
fi

echo "== left in $RG"
az resource list -g "$RG" --query "[].{name:name, type:type}" -o table
