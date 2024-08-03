source ./scripts/server_config.sh

for service in "${!SERVICES[@]}"; do
    start_service "$service" "${SERVICES[$service]}"
done