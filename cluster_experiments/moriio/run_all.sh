#!/bin/bash
# Launch full 1P1D MoRIIO PD-disagg stack and run a smoke test
# Usage: NODE_IP=<ip> MODEL=<path> bash run_all.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NODE_IP=${NODE_IP:-$(hostname -I | awk '{print $1}')}
export MODEL=${MODEL:-/shared_inference/models/Qwen/Qwen3-8B}
export TP=${TP:-4}
export PREFILL_GPUS=${PREFILL_GPUS:-0,1,2,3}
export DECODE_GPUS=${DECODE_GPUS:-4,5,6,7}
export PREFILL_PORT=${PREFILL_PORT:-20005}
export DECODE_PORT=${DECODE_PORT:-40005}
export DISCOVERY_PORT=${DISCOVERY_PORT:-36367}
export ROUTER_PORT=${ROUTER_PORT:-30000}
export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai-rocm:v0.23.0}

MODEL_NAME=$(basename "$MODEL")

echo "========================================"
echo "  Orbit MoRIIO PD-Disagg Stack"
echo "  NODE_IP:  $NODE_IP"
echo "  MODEL:    $MODEL_NAME"
echo "  TP:       $TP"
echo "  Prefill:  GPUs=$PREFILL_GPUS  port=$PREFILL_PORT"
echo "  Decode:   GPUs=$DECODE_GPUS   port=$DECODE_PORT"
echo "  Router:   port=$ROUTER_PORT"
echo "========================================"

# Clean up old containers
echo "[*] Cleaning up old containers..."
docker rm -f orbit_router orbit_prefill orbit_decode 2>/dev/null || true
sleep 2

# Start router first (discovery proxy)
bash "${SCRIPT_DIR}/start_router.sh"

# Start prefill and decode
bash "${SCRIPT_DIR}/start_prefill.sh"
bash "${SCRIPT_DIR}/start_decode.sh"

echo ""
echo "[*] Waiting for model load (120s)..."
sleep 360

echo ""
echo "=== Health checks ==="
for name_port in "router:$ROUTER_PORT" "prefill:$PREFILL_PORT" "decode:$DECODE_PORT"; do
    name="${name_port%%:*}"
    port="${name_port##*:}"
    if curl -sf "http://127.0.0.1:${port}/health" > /dev/null 2>&1; then
        echo "  $name (port $port): HEALTHY"
    else
        echo "  $name (port $port): FAILED"
        docker logs --tail=20 "orbit_${name}" 2>&1 | sed "s/^/    [$name] /"
    fi
done

echo ""
echo "=== Smoke test: What is the capital of France? ==="
curl -s -m 60 http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
        \"max_tokens\": 20
    }" | python3 -m json.tool 2>/dev/null || echo "(raw response above)"

echo ""
echo "Done. Logs: docker logs -f orbit_prefill / orbit_decode / orbit_router"
