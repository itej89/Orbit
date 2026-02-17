#!/bin/bash
# =============================================================================
# Start Orbit Router (MPC-enhanced)
# Usage: ./start_orbit_router.sh [config]
# =============================================================================

set -e

CONFIG=${1:-"2p2d"}
ROUTING_MODE=${ROUTING_MODE:-"po2"}
ENABLE_MPC=${ENABLE_MPC:-"true"}
ROUTER_PORT=${ROUTER_PORT:-8000}

# Server configurations
case $CONFIG in
    "2p2d")
        PREFILL_HOSTS="127.0.0.1 127.0.0.1"
        PREFILL_PORTS="8100 8101"
        DECODE_HOSTS="127.0.0.1 127.0.0.1"
        DECODE_PORTS="8200 8201"
        ;;
    "4p4d")
        PREFILL_HOSTS="127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1"
        PREFILL_PORTS="8100 8101 8102 8103"
        DECODE_HOSTS="127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1"
        DECODE_PORTS="8200 8201 8202 8203"
        ;;
    "8p8d")
        PREFILL_HOSTS="127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1"
        PREFILL_PORTS="8100 8101 8102 8103 8104 8105 8106 8107"
        DECODE_HOSTS="127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1"
        DECODE_PORTS="8200 8201 8202 8203 8204 8205 8206 8207"
        ;;
    "custom")
        # Use environment variables
        if [[ -z "$PREFILL_HOSTS" || -z "$DECODE_HOSTS" ]]; then
            echo "For custom config, set PREFILL_HOSTS, PREFILL_PORTS, DECODE_HOSTS, DECODE_PORTS"
            exit 1
        fi
        ;;
    *)
        echo "Unknown config: $CONFIG"
        echo "Available: 2p2d, 4p4d, 8p8d, custom"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "Starting Orbit Router"
echo "Config: $CONFIG"
echo "Routing Mode: $ROUTING_MODE"
echo "MPC Enabled: $ENABLE_MPC"
echo "Router Port: $ROUTER_PORT"
echo "=============================================="

# Build command
CMD="python $PROJECT_ROOT/src/orbit/router.py"
CMD="$CMD --prefiller-hosts $PREFILL_HOSTS"
CMD="$CMD --prefiller-ports $PREFILL_PORTS"
CMD="$CMD --decoder-hosts $DECODE_HOSTS"
CMD="$CMD --decoder-ports $DECODE_PORTS"
CMD="$CMD --host 0.0.0.0"
CMD="$CMD --port $ROUTER_PORT"

echo ""
echo "Command: $CMD"
echo ""

# Run
exec $CMD
