#!/bin/bash
# Start vllm-router with MoRIIOConnector discovery
set -e

DISCOVERY_PORT=${DISCOVERY_PORT:-36367}
ROUTER_PORT=${ROUTER_PORT:-30000}

echo "[router] Stopping any existing router container..."
docker rm -f orbit_router 2>/dev/null || true

echo "[router] Starting vllm-router on port $ROUTER_PORT (discovery: $DISCOVERY_PORT)..."
docker run -d --name orbit_router \
    --network host \
    vllm/vllm-router:nightly \
    vllm-router \
    --vllm-pd-disaggregation \
    --kv-connector moriio \
    --vllm-discovery-address "0.0.0.0:${DISCOVERY_PORT}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}"

echo "[router] Started. Waiting 5s..."
sleep 5
echo "[router] Done."
