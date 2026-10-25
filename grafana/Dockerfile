FROM grafana/grafana:13.2.4

ENV GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_AUTH_DISABLE_LOGIN_FORM=true \
    GF_AZURE_MANAGED_IDENTITY_ENABLED=true \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/robotics-azure.json

COPY provisioning /etc/grafana/provisioning
ARG SUBSCRIPTION_ID
COPY dashboards /var/lib/grafana/dashboards
USER root
RUN test -n "$SUBSCRIPTION_ID" && sed -i "s/__SUBSCRIPTION_ID__/$SUBSCRIPTION_ID/g" /var/lib/grafana/dashboards/*.json
USER grafana
