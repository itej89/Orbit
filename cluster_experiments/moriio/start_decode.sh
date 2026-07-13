#!/bin/bash
# Start vLLM decode (kv_consumer) with MoRIIOConnector
set -e

NODE_IP=${NODE_IP:-$(hostname -I | awk '{print $1}')}
MODEL=${MODEL:-/shared_inference/models/Qwen/Qwen3-8B}
PORT=${DECODE_PORT:-40005}
TP=${TP:-4}
GPUS=${DECODE_GPUS:-4,5,6,7}
HANDSHAKE_PORT=${DECODE_HANDSHAKE_PORT:-7301}
NOTIFY_PORT=${DECODE_NOTIFY_PORT:-7501}
DISCOVERY_PORT=${DISCOVERY_PORT:-36367}
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai-rocm:v0.23.0}
GPU_MEM=${GPU_MEM:-0.85}

echo "[decode] Stopping any existing decode container..."
docker rm -f orbit_decode 2>/dev/null || true

echo "[decode] Starting on GPUs $GPUS, port $PORT, model $(basename $MODEL)..."
docker run -d --name orbit_decode \
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
    -e VLLM_ROCM_USE_AITER=1 \
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
            \"kv_role\": \"kv_consumer\",
            \"kv_connector_extra_config\": {
                \"proxy_ip\": \"${NODE_IP}\",
                \"http_port\": \"${PORT}\",
                \"proxy_ping_port\": \"${DISCOVERY_PORT}\",
                \"handshake_port\": \"${HANDSHAKE_PORT}\",
                \"notify_port\": \"${NOTIFY_PORT}\"
            }
        }"

echo "[decode] Started. Container: orbit_decode"
