#!/bin/bash
# Start vLLM prefill (kv_producer) with MoRIIOConnector
set -e

NODE_IP=${NODE_IP:-$(hostname -I | awk '{print $1}')}
MODEL=${MODEL:-/shared_inference/models/Qwen/Qwen3-8B}
PORT=${PREFILL_PORT:-20005}
TP=${TP:-4}
GPUS=${PREFILL_GPUS:-0,1,2,3}
HANDSHAKE_PORT=${PREFILL_HANDSHAKE_PORT:-6301}
NOTIFY_PORT=${PREFILL_NOTIFY_PORT:-6105}
DISCOVERY_PORT=${DISCOVERY_PORT:-36367}
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai-rocm:v0.23.0}
GPU_MEM=${GPU_MEM:-0.85}

echo "[prefill] Stopping any existing prefill container..."
docker rm -f orbit_prefill 2>/dev/null || true

echo "[prefill] Starting on GPUs $GPUS, port $PORT, model $(basename $MODEL)..."
docker run -d --name orbit_prefill \
    --user "$(id -u):$(id -g)" \
    --device /dev/dri \
    --device /dev/kfd \
    --device /dev/infiniband \
    --network host \
    --ipc host \
    --group-add "$(getent group video | cut -d: -f3)" \
    --group-add "$(getent group render | cut -d: -f3)" \
    --cap-add SYS_PTRACE \
    --cap-add IPC_LOCK \
    --security-opt seccomp=unconfined \
    --shm-size 64G \
    --ulimit nofile=1048576:1048576 \
    --ulimit memlock=-1:-1 \
    -v /shared_inference:/shared_inference \
    -e HOME=/shared_inference/vpolamre \
    -e USER="$(id -un)" \
    -e VLLM_ROCM_USE_AITER=0 \
    -e AITER_JIT_DIR=/tmp/aiter_jit_prefill \
    -e TRITON_CACHE_DIR=/tmp/triton_prefill \
    -e VLLM_CACHE_ROOT=/tmp/vllm_prefill \
    -e VLLM_LOGGING_LEVEL=DEBUG \
    -e MORI_RDMA_DEVICES=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9 \
    -e MORI_IB_GID_INDEX=3 \
    -e MORI_SHMEM_HEAP_SIZE=16G \
    -e MORI_GPU_ARCHS=gfx942 \
    -e NCCL_IB_GID_INDEX=3 \
    -e HIP_VISIBLE_DEVICES="${GPUS}" \
    "${IMAGE}" \
    "${MODEL}" \
        --tensor-parallel-size "${TP}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEM}" \
        --kv-transfer-config "{
            \"kv_connector\": \"MoRIIOConnector\",
            \"kv_role\": \"kv_producer\",
            \"kv_connector_extra_config\": {
                \"proxy_ip\": \"${NODE_IP}\",
                \"proxy_ping_port\": \"${DISCOVERY_PORT}\",
                \"http_port\": \"${PORT}\",
                \"handshake_port\": \"${HANDSHAKE_PORT}\",
                \"notify_port\": \"${NOTIFY_PORT}\"
            }
        }"

echo "[prefill] Started. Container: orbit_prefill"
