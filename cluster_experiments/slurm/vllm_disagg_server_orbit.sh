#!/bin/bash
# =============================================================================
# VLLM Disaggregated Server with Orbit MPC Proxy and Heterogeneous Support
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"

# Routing policy: toy (default vllm proxy), rr, random, po2, mpc_rr, mpc_po2
PROXY_POLICY="${PROXY_POLICY:-toy}"

# Heterogeneous mode: throttle even-ranked P/D nodes
HETERO_MODE="${HETERO_MODE:-false}"
HETERO_GPU_MEM="${HETERO_GPU_MEM:-0.45}"
HETERO_MAX_SEQS="${HETERO_MAX_SEQS:-64}"
DEFAULT_GPU_MEM="${DEFAULT_GPU_MEM:-0.7}"
DEFAULT_MAX_SEQS="${DEFAULT_MAX_SEQS:-256}"

echo "=== Orbit Disaggregated Server ==="
echo "Policy: $PROXY_POLICY | Hetero: $HETERO_MODE"
echo "Listing NIXL_COOKBOOK_PATH:"
ls ${NIXL_COOKBOOK_PATH}

# =============================================================================
# Dependencies
# =============================================================================

pip install py-spy 2>/dev/null
pip install --ignore-installed --force-reinstall flask 2>/dev/null

if [[ "$PROXY_POLICY" == mpc_* ]] && [[ "$NODE_RANK" == "0" ]]; then
    pip install cvxpy numpy 2>/dev/null
fi

host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

# =============================================================================
# Dynamic Port Allocation
# =============================================================================

DEFAULT_BARRIER_PORT=5000
DEFAULT_ETCD_CLIENT_PORT=2379
DEFAULT_ETCD_PEER_PORT=2380
DEFAULT_VLLM_PORT=2584
DEFAULT_NIXL_PORT=14600

PORT_CONFIG_DIR="${PORT_CONFIG_DIR:-/port_config}"
PORT_CONFIG_FILE="${PORT_CONFIG_DIR}/${SLURM_JOB_ID}_port_config.txt"
mkdir -p "${PORT_CONFIG_DIR}" 2>/dev/null || true

is_port_available() {
    local port=$1
    python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', $port))
    s.close()
    exit(0)
except:
    exit(1)
" 2>/dev/null
    return $?
}

find_port_offset() {
    local increment=10
    local max_attempts=10
    for ((attempt=0; attempt<max_attempts; attempt++)); do
        local offset=$((attempt * increment))
        local all_available=true
        for base_port in $DEFAULT_BARRIER_PORT $DEFAULT_ETCD_CLIENT_PORT $DEFAULT_ETCD_PEER_PORT $DEFAULT_VLLM_PORT; do
            local test_port=$((base_port + offset))
            if ! is_port_available $test_port; then
                all_available=false
                break
            fi
        done
        if $all_available; then
            echo $offset
            return 0
        fi
    done
    echo "0"
    return 1
}

allocate_ports() {
    # Use job-ID-based offset to avoid collisions between sequential jobs
    # Each job gets a deterministic unique offset (range 0-190, step 10)
    PORT_OFFSET=$(( (SLURM_JOB_ID % 20) * 10 ))
    echo "Job ${SLURM_JOB_ID}: Using deterministic port offset $PORT_OFFSET"

    # Verify primary port is available, fall back to dynamic search if not
    local test_port=$((DEFAULT_VLLM_PORT + PORT_OFFSET))
    if ! is_port_available $test_port; then
        echo "Port $test_port busy, finding alternative..."
        PORT_OFFSET=$(find_port_offset)
        echo "Fallback port offset: $PORT_OFFSET"
    fi

    export BARRIER_PORT=$((DEFAULT_BARRIER_PORT + PORT_OFFSET))
    export ETCD_CLIENT_PORT=$((DEFAULT_ETCD_CLIENT_PORT + PORT_OFFSET))
    export ETCD_PEER_PORT=$((DEFAULT_ETCD_PEER_PORT + PORT_OFFSET))
    export SERVER_PORT=$((DEFAULT_VLLM_PORT + PORT_OFFSET))
    export NIXL_PORT=$((DEFAULT_NIXL_PORT + PORT_OFFSET))

    echo "=== Port Configuration ==="
    echo "  Barrier:     $BARRIER_PORT"
    echo "  ETCD Client: $ETCD_CLIENT_PORT"
    echo "  ETCD Peer:   $ETCD_PEER_PORT"
    echo "  vLLM:        $SERVER_PORT"
    echo "  NIXL:        $NIXL_PORT"
}

BARRIER_PORT=$DEFAULT_BARRIER_PORT
ETCD_CLIENT_PORT=$DEFAULT_ETCD_CLIENT_PORT
ETCD_PEER_PORT=$DEFAULT_ETCD_PEER_PORT
SERVER_PORT=$DEFAULT_VLLM_PORT
NIXL_PORT=$DEFAULT_NIXL_PORT

# =============================================================================
# Model Configuration
# =============================================================================

declare -A MODEL_PREFILL_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
    ["Qwen14B"]="--tensor-parallel-size 8 --max-model-len 8192"
    ["Qwen32B"]="--tensor-parallel-size 8 --max-model-len 8192"
)

declare -A MODEL_DECODE_CONFIGS=(
    ["Llama-3.1-405B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --kv-cache-dtype fp8"
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="--tensor-parallel-size 8 --max-model-len 65536 --kv-cache-dtype fp8"
    ["DeepSeek-V3"]="--tensor-parallel-size 8 --compilation-config '{\"cudagraph_mode\":\"PIECEWISE\"}' --no-enable-prefix-caching --block-size 1"
    ["gpt-oss-120b"]="--tensor-parallel-size 8"
    ["Qwen14B"]="--tensor-parallel-size 8 --max-model-len 8192"
    ["Qwen32B"]="--tensor-parallel-size 8 --max-model-len 8192"
)

declare -A MODEL_ENVS=(
    ["amd-Llama-3.3-70B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 "
    ["Llama-3.1-405B-Instruct-FP8-KV"]="VLLM_USE_V1=1 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 AMDGCN_USE_BUFFER_OPS=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_ROPE=1 TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1 TRITON_HIP_USE_ASYNC_COPY=1 TRITON_HIP_USE_BLOCK_PINGPONG=1 TRITON_HIP_ASYNC_FAST_SWIZZLE=1 "
    ["DeepSeek-V3"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_PAGED_ATTN=0 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_USE_AITER_TRITON_SILU_MUL=0 "
    ["gpt-oss-120b"]="VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_TRITON_BF16_GEMM=0 VLLM_USE_AITER_UNIFIED_ATTENTION=1 VLLM_ROCM_USE_AITER_MHA=0 ROCM_TRITON_MOE_PRESHUFFLE_SCALES=0 "
)

get_model_config() {
    local mode="$1"
    local model_name="$2"
    if [[ "$mode" == "prefill" ]]; then
        echo "${MODEL_PREFILL_CONFIGS[$model_name]:-"--tensor-parallel-size 4"}"
    else
        echo "${MODEL_DECODE_CONFIGS[$model_name]:-"--tensor-parallel-size 4"}"
    fi
}

get_model_envs() {
    echo "${MODEL_ENVS[$1]:-""}"
}

if [[ -z "$MODEL_NAME" ]]; then
    echo "ERROR: MODEL_NAME not set"
    exit 1
fi

PREFILL_MODEL_CONFIG=$(get_model_config "prefill" "$MODEL_NAME")
DECODE_MODEL_CONFIG=$(get_model_config "decode" "$MODEL_NAME")
PREFILL_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
DECODE_MODEL_ENVS=$(get_model_envs "$MODEL_NAME")
echo "Model: $MODEL_NAME | Prefill cfg: $PREFILL_MODEL_CONFIG"

# =============================================================================
# Port Allocation
# =============================================================================

echo "Allocating ports..."
allocate_ports

# =============================================================================
# Determine heterogeneous settings for this node
# =============================================================================

NODE_GPU_MEM="$DEFAULT_GPU_MEM"
NODE_MAX_SEQS="$DEFAULT_MAX_SEQS"

if [[ "$HETERO_MODE" == "true" ]]; then
    # In heterogeneous mode, even-ranked P/D nodes (relative to their role) get throttled
    # For 2P2D: nodes 1,2 = prefill, nodes 3,4 = decode
    # Node 2 (second prefill) and Node 4 (second decode) get throttled
    if [[ "$NODE_RANK" -gt 0 ]]; then
        if [[ "$NODE_RANK" -le "$xP" ]]; then
            # Prefill node - throttle if it's an even-indexed prefill
            local_rank=$((NODE_RANK))
            if [[ $((local_rank % 2)) -eq 0 ]]; then
                NODE_GPU_MEM="$HETERO_GPU_MEM"
                NODE_MAX_SEQS="$HETERO_MAX_SEQS"
                echo "*** HETEROGENEOUS: Prefill node $NODE_RANK throttled (gpu_mem=$NODE_GPU_MEM, max_seqs=$NODE_MAX_SEQS) ***"
            fi
        else
            # Decode node - throttle if it's an even-indexed decode
            local_rank=$((NODE_RANK - xP))
            if [[ $((local_rank % 2)) -eq 0 ]]; then
                NODE_GPU_MEM="$HETERO_GPU_MEM"
                NODE_MAX_SEQS="$HETERO_MAX_SEQS"
                echo "*** HETEROGENEOUS: Decode node $NODE_RANK throttled (gpu_mem=$NODE_GPU_MEM, max_seqs=$NODE_MAX_SEQS) ***"
            fi
        fi
    fi
fi

# =============================================================================
# Container Barrier
# =============================================================================

echo "Waiting at container creation barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port ${BARRIER_PORT} \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports ${BARRIER_PORT}

# =============================================================================
# ETCD Setup
# =============================================================================

echo "Starting etcd on $host_name"
export ETCD_CLIENT_PORT
export ETCD_PEER_PORT

${NIXL_COOKBOOK_PATH}/start_etcd.sh > /dev/null &
etcd_pid=$!

python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --node-ips ${IPADDRS} \
    --node-ports ${ETCD_CLIENT_PORT}

echo "All etcd servers up: $host_name"
sleep 3

export ETCDCTL_ENDPOINTS="127.0.0.1:${ETCD_CLIENT_PORT}"
/usr/local/bin/etcd//etcdctl endpoint health 2>&1 || true

python $NIXL_COOKBOOK_PATH/socket_barrier.py --node-ips ${IPADDRS} --node-ports ${ETCD_CLIENT_PORT}

# =============================================================================
# Cluster Topology
# =============================================================================

IFS=',' read -ra IP_ARRAY <<< "$IPADDRS"

PREFILL_ARGS=""
DECODE_ARGS=""
PREFILL_PORTS=""
DECODE_PORTS=""

for ((i=1; i<=$xP && i<${#IP_ARRAY[@]}; i++)); do
    PREFILL_ARGS+="${IP_ARRAY[$i]} "
    PREFILL_PORTS+="$SERVER_PORT "
done

for ((i=xP+1; i<${#IP_ARRAY[@]}; i++)); do
    DECODE_ARGS+="${IP_ARRAY[$i]} "
    DECODE_PORTS+="$SERVER_PORT "
done

# =============================================================================
# Node Role: Proxy (NODE_RANK=0)
# =============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== PROXY NODE: ${host_name}:${host_ip} ==="
    echo "  Prefill: ${PREFILL_ARGS}"
    echo "  Decode:  ${DECODE_ARGS}"
    echo "  Policy:  ${PROXY_POLICY}"

    PD_IPADDRS="${IPADDRS#*,}"
    echo "Waiting for P/D servers..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${PD_IPADDRS} \
        --node-ports $SERVER_PORT

    if [[ "$PROXY_POLICY" == "toy" ]] || [[ "$PROXY_POLICY" == "rr_toy" ]]; then
        # Use the built-in toy_proxy_server (round-robin only)
        UCX_TLS=tcp,self,shm UCX_NET_DEVICES=ens11np0 NCCL_UCX_TLS=tcp NCCL_SOCKET_IFNAME=ens11np0 VLLM_USE_V1=1 \
        python3 "/app/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
                --host 0.0.0.0 \
                --port $SERVER_PORT \
                --prefiller-hosts ${PREFILL_ARGS} \
                --prefiller-ports ${PREFILL_PORTS} \
                --decoder-hosts ${DECODE_ARGS} \
                --decoder-ports ${DECODE_PORTS} \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
    else
        # Use orbit_proxy_server.py with configurable policy
        UCX_TLS=tcp,self,shm UCX_NET_DEVICES=ens11np0 NCCL_UCX_TLS=tcp NCCL_SOCKET_IFNAME=ens11np0 VLLM_USE_V1=1 \
        python3 /opt/orbit/orbit_proxy_server.py \
                --host 0.0.0.0 \
                --port $SERVER_PORT \
                --prefiller-hosts ${PREFILL_ARGS} \
                --prefiller-ports ${PREFILL_PORTS} \
                --decoder-hosts ${DECODE_ARGS} \
                --decoder-ports ${DECODE_PORTS} \
                --policy ${PROXY_POLICY} \
                2>&1 | tee /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &
        proxy_pid=$!
    fi

    echo "Waiting for proxy to be ready..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${host_ip} \
        --node-ports $SERVER_PORT

    echo "Proxy ready on ${host_name}:${host_ip}:${SERVER_PORT}"
    sleep 10

    SERVER_PORT=$SERVER_PORT bash $NIXL_COOKBOOK_PATH/benchmark_xPyD.sh

    echo "Killing proxy server"
    kill $proxy_pid

# =============================================================================
# Node Role: Prefill (NODE_RANK 1..xP)
# =============================================================================

elif [ "$NODE_RANK" -gt 0 ] && [ "$NODE_RANK" -le "$xP" ]; then
    echo "=== PREFILL NODE ${NODE_RANK}: ${host_name}:${host_ip} ==="
    echo "  GPU Mem: ${NODE_GPU_MEM} | Max Seqs: ${NODE_MAX_SEQS}"

    PREFILL_CMD="LD_LIBRARY_PATH=/app/install/nixl/lib/x86_64-linux-gnu/:/app/install/ucx/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH \
    ${PREFILL_MODEL_ENVS} \
    VLLM_USE_V1=1 \
    VLLM_SERVER_DEV_MODE=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=\${host_ip} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
    UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp \
    UCX_NET_DEVICES=mlx5_0:1 \
    UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp \
    UCX_SOCKADDR_CM_ENABLE=y \
    UCX_RDMA_CM_ENABLED=y \
    UCX_MEMTYPE_CACHE=y \
    UCX_RNDV_SCHEME=get_zcopy \
    UCX_RNDV_THRESH=4k \
    UCX_ROCM_IPC_MIN_ZCOPY=0 \
    HSA_ENABLE_SDMA=1 \
    UCX_LOG_LEVEL=info \
    NIXL_LOG_LEVEL=DEBUG \
    vllm serve \${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization ${NODE_GPU_MEM} \
        --max-num-seqs ${NODE_MAX_SEQS} \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"pd-run\", \"kv_role\": \"kv_producer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${host_ip}\"'\", \"kv_port\": '\"${NIXL_PORT}\"'}'"

    if [[ -n "$PREFILL_MODEL_CONFIG" ]]; then
        PREFILL_CMD="$PREFILL_CMD $PREFILL_MODEL_CONFIG"
    fi

    eval "$PREFILL_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log >/dev/null &
    prefill_pid=$!

    echo "Waiting for proxy..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $SERVER_PORT

    echo "Waiting until proxy closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $SERVER_PORT

    echo "Killing prefill server"
    kill $prefill_pid

# =============================================================================
# Node Role: Decode (NODE_RANK > xP)
# =============================================================================

else
    echo "=== DECODE NODE ${NODE_RANK}: ${host_name}:${host_ip} ==="
    echo "  GPU Mem: ${NODE_GPU_MEM} | Max Seqs: ${NODE_MAX_SEQS}"

    DECODE_CMD="LD_LIBRARY_PATH=/app/install/nixl/lib/x86_64-linux-gnu/:/app/install/ucx/lib:/opt/rocm/lib:\$LD_LIBRARY_PATH \
    ${DECODE_MODEL_ENVS} \
    VLLM_USE_V1=1 \
    VLLM_SERVER_DEV_MODE=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=\${host_ip} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5557 \
    UCX_TLS=rc,sm,self,rocm_copy,rocm_ipc,tcp \
    UCX_NET_DEVICES=mlx5_0:1 \
    UCX_SOCKADDR_TLS_PRIORITY=rdmacm,tcp \
    UCX_SOCKADDR_CM_ENABLE=y \
    UCX_RDMA_CM_ENABLED=y \
    UCX_MEMTYPE_CACHE=y \
    UCX_RNDV_SCHEME=get_zcopy \
    UCX_RNDV_THRESH=4k \
    UCX_ROCM_IPC_MIN_ZCOPY=0 \
    HSA_ENABLE_SDMA=1 \
    UCX_LOG_LEVEL=info \
    NIXL_LOG_LEVEL=DEBUG \
    vllm serve \${MODEL_PATH} \
        --port $SERVER_PORT \
        --trust-remote-code \
        --disable-log-requests \
        --gpu-memory-utilization ${NODE_GPU_MEM} \
        --max-num-seqs ${NODE_MAX_SEQS} \
        --kv-transfer-config '{\"kv_connector\": \"NixlConnector\", \"engine_id\": \"llama8b-run\", \"kv_role\": \"kv_consumer\", \"kv_parallel_size\": 8, \"kv_rank\": 0, \"kv_buffer_size\": 5000000000, \"kv_buffer_device\": \"cuda\", \"kv_ip\": \"'\"\${host_ip}\"'\", \"kv_port\": '\"${NIXL_PORT}\"'}'"

    if [[ -n "$DECODE_MODEL_CONFIG" ]]; then
        DECODE_CMD="$DECODE_CMD $DECODE_MODEL_CONFIG"
    fi

    eval "$DECODE_CMD" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log >/dev/null &
    decode_pid=$!

    echo "Waiting for proxy..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $SERVER_PORT

    echo "Waiting until proxy closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $SERVER_PORT

    echo "Killing decode server"
    kill $decode_pid
fi

echo "Killing etcd server"
kill $etcd_pid

echo "Script completed successfully"
exit 0
