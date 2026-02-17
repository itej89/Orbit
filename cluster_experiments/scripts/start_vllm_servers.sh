#!/bin/bash
# =============================================================================
# Start vLLM Disaggregated Servers
# Usage: ./start_vllm_servers.sh <config_name>
# =============================================================================

set -e

CONFIG_NAME=${1:-"het_2p2d_a"}
MODEL=${MODEL:-"meta-llama/Llama-3.1-70B-Instruct"}
BASE_IMAGE="rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta"

# Configuration definitions
case $CONFIG_NAME in
    # Homogeneous: all TP=8
    "homo_2p2d_tp8")
        PREFILL_CONFIGS=("TP=8:GPU=0-7:PORT=8100" "TP=8:GPU=0-7:PORT=8101")
        DECODE_CONFIGS=("TP=8:GPU=0-7:PORT=8200" "TP=8:GPU=0-7:PORT=8201")
        ;;
    
    # Heterogeneous: P1=TP8, P2=TP4, D1=TP8, D2=TP4
    "het_2p2d_a")
        PREFILL_CONFIGS=("TP=8:GPU=0-7:PORT=8100" "TP=4:GPU=0-3:PORT=8101")
        DECODE_CONFIGS=("TP=8:GPU=0-7:PORT=8200" "TP=4:GPU=0-3:PORT=8201")
        ;;
    
    # Heterogeneous: P1=TP8, P2=TP2, D1=TP8, D2=TP2 (4x diff)
    "het_2p2d_b")
        PREFILL_CONFIGS=("TP=8:GPU=0-7:PORT=8100" "TP=2:GPU=0-1:PORT=8101")
        DECODE_CONFIGS=("TP=8:GPU=0-7:PORT=8200" "TP=2:GPU=0-1:PORT=8201")
        ;;
    
    # 4P4D mixed
    "het_4p4d")
        PREFILL_CONFIGS=(
            "TP=8:GPU=0-7:PORT=8100"
            "TP=8:GPU=0-7:PORT=8101"
            "TP=4:GPU=0-3:PORT=8102"
            "TP=4:GPU=0-3:PORT=8103"
        )
        DECODE_CONFIGS=(
            "TP=8:GPU=0-7:PORT=8200"
            "TP=8:GPU=0-7:PORT=8201"
            "TP=4:GPU=0-3:PORT=8202"
            "TP=4:GPU=0-3:PORT=8203"
        )
        ;;
    
    *)
        echo "Unknown config: $CONFIG_NAME"
        echo "Available: homo_2p2d_tp8, het_2p2d_a, het_2p2d_b, het_4p4d"
        exit 1
        ;;
esac

# Parse config string: "TP=N:GPU=X-Y:PORT=P"
parse_config() {
    local config=$1
    local tp=$(echo $config | grep -oP 'TP=\K[0-9]+')
    local gpu=$(echo $config | grep -oP 'GPU=\K[0-9-]+')
    local port=$(echo $config | grep -oP 'PORT=\K[0-9]+')
    echo "$tp $gpu $port"
}

# Convert GPU range to HIP_VISIBLE_DEVICES format
gpu_range_to_list() {
    local range=$1
    if [[ $range == *"-"* ]]; then
        local start=$(echo $range | cut -d'-' -f1)
        local end=$(echo $range | cut -d'-' -f2)
        seq -s',' $start $end
    else
        echo $range
    fi
}

echo "=============================================="
echo "Starting vLLM Disaggregated Servers"
echo "Config: $CONFIG_NAME"
echo "Model: $MODEL"
echo "=============================================="

# Start prefill servers
echo ""
echo "--- Starting Prefill Servers ---"
for i in "${!PREFILL_CONFIGS[@]}"; do
    read tp gpu port <<< $(parse_config "${PREFILL_CONFIGS[$i]}")
    gpu_list=$(gpu_range_to_list $gpu)
    
    echo "Prefill Server $i: TP=$tp, GPUs=$gpu_list, Port=$port"
    
    # Note: In actual cluster, run this via job scheduler
    cat << EOF
# Command for Prefill Server $i:
HIP_VISIBLE_DEVICES=$gpu_list docker run -d \\
    --network=host \\
    --device=/dev/kfd --device=/dev/dri \\
    --group-add video --ipc=host --shm-size=64GB \\
    --name prefill_server_$i \\
    $BASE_IMAGE \\
    python -m vllm.entrypoints.openai.api_server \\
        --model $MODEL \\
        --tensor-parallel-size $tp \\
        --port $port \\
        --kv-transfer-config '{"kv_connector": "PyNcclConnector"}' \\
        --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \\
        --no-enable-prefix-caching

EOF
done

# Start decode servers
echo ""
echo "--- Starting Decode Servers ---"
for i in "${!DECODE_CONFIGS[@]}"; do
    read tp gpu port <<< $(parse_config "${DECODE_CONFIGS[$i]}")
    gpu_list=$(gpu_range_to_list $gpu)
    
    echo "Decode Server $i: TP=$tp, GPUs=$gpu_list, Port=$port"
    
    cat << EOF
# Command for Decode Server $i:
HIP_VISIBLE_DEVICES=$gpu_list docker run -d \\
    --network=host \\
    --device=/dev/kfd --device=/dev/dri \\
    --group-add video --ipc=host --shm-size=64GB \\
    --name decode_server_$i \\
    $BASE_IMAGE \\
    python -m vllm.entrypoints.openai.api_server \\
        --model $MODEL \\
        --tensor-parallel-size $tp \\
        --port $port \\
        --kv-transfer-config '{"kv_connector": "PyNcclConnector"}' \\
        --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \\
        --no-enable-prefix-caching

EOF
done

echo ""
echo "=============================================="
echo "Server startup commands generated."
echo "Copy and run on appropriate compute nodes."
echo "=============================================="
