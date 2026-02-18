#!/bin/bash
#
# Full Orbit MPC Experiment Runner
# 
# Runs experiments on vLLM disaggregated serving with MPC-augmented routing
# across multiple models and heterogeneous server configurations.
#
# Usage:
#   ./run_full_experiment.sh [--model MODEL] [--config CONFIG] [--quick]
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/cluster_experiments/results/$(date +%Y%m%d_%H%M%S)"

# Default vLLM image
VLLM_IMAGE="${VLLM_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta}"

# Cluster settings
CLUSTER_HOST="${CLUSTER_HOST:-useocpslog-002.amd.com}"
GPUS_PER_NODE=8

# Models to test
MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
)

# Server configurations: config_name:prefill_tp:decode_tp
CONFIGS=(
    "homo_8x8:8:8"
    "het_8x4:8:4"
    "het_8x2:8:2"
)

# =============================================================================
# Parse Arguments
# =============================================================================

QUICK_MODE=false
SELECTED_MODEL=""
SELECTED_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            SELECTED_MODEL="$2"
            shift 2
            ;;
        --config)
            SELECTED_CONFIG="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --image)
            VLLM_IMAGE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    log "Cleaning up containers..."
    docker ps -a --filter "name=orbit_" -q | xargs -r docker rm -f 2>/dev/null || true
}

trap cleanup EXIT

get_gpu_list() {
    local start=$1
    local count=$2
    local gpus=""
    for ((i=start; i<start+count; i++)); do
        if [ -n "$gpus" ]; then
            gpus="$gpus,"
        fi
        gpus="$gpus$i"
    done
    echo "$gpus"
}

# =============================================================================
# Start vLLM Servers
# =============================================================================

start_prefill_server() {
    local name=$1
    local model=$2
    local tp=$3
    local port=$4
    local gpu_start=$5
    
    local gpu_list=$(get_gpu_list $gpu_start $tp)
    
    log "Starting prefill server $name (TP=$tp, GPUs=$gpu_list, port=$port)"
    
    docker run -d \
        --name "orbit_prefill_$name" \
        --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=64GB \
        -e HIP_VISIBLE_DEVICES="$gpu_list" \
        "$VLLM_IMAGE" \
        python -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --tensor-parallel-size $tp \
            --port $port \
            --kv-transfer-config '{"kv_connector": "PyNcclConnector"}' \
            --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
            --no-enable-prefix-caching
    
    # Wait for server to start
    log "Waiting for prefill server $name to be ready..."
    for i in {1..120}; do
        if curl -s "http://localhost:$port/health" | grep -q "ok"; then
            log "Prefill server $name is ready"
            return 0
        fi
        sleep 5
    done
    
    log "ERROR: Prefill server $name failed to start"
    return 1
}

start_decode_server() {
    local name=$1
    local model=$2
    local tp=$3
    local port=$4
    local gpu_start=$5
    
    local gpu_list=$(get_gpu_list $gpu_start $tp)
    
    log "Starting decode server $name (TP=$tp, GPUs=$gpu_list, port=$port)"
    
    docker run -d \
        --name "orbit_decode_$name" \
        --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=64GB \
        -e HIP_VISIBLE_DEVICES="$gpu_list" \
        "$VLLM_IMAGE" \
        python -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --tensor-parallel-size $tp \
            --port $port \
            --kv-transfer-config '{"kv_connector": "PyNcclConnector"}' \
            --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
    
    # Wait for server to start
    log "Waiting for decode server $name to be ready..."
    for i in {1..120}; do
        if curl -s "http://localhost:$port/health" | grep -q "ok"; then
            log "Decode server $name is ready"
            return 0
        fi
        sleep 5
    done
    
    log "ERROR: Decode server $name failed to start"
    return 1
}

# =============================================================================
# Start Orbit Router
# =============================================================================

start_orbit_router() {
    local prefill_ports=$1
    local decode_ports=$2
    local router_port=$3
    local mpc_enabled=$4
    
    log "Starting Orbit router (MPC=$mpc_enabled)"
    
    local mpc_flag=""
    if [ "$mpc_enabled" = "true" ]; then
        # Start MPC controller first
        docker run -d \
            --name "orbit_mpc_controller" \
            --network=host \
            -v "$PROJECT_ROOT:/workspace" \
            "$VLLM_IMAGE" \
            python /workspace/vllm_router_fork/scripts/mpc_controller.py \
                --port 8090 \
                --router-metrics-url "http://localhost:$router_port/metrics"
        
        sleep 5
        mpc_flag="--mpc-enabled --mpc-controller-url http://localhost:8090"
    fi
    
    # Start router using our simulation router for now
    # TODO: Switch to actual vllm-router binary when built
    docker run -d \
        --name "orbit_router" \
        --network=host \
        -v "$PROJECT_ROOT:/workspace" \
        "$VLLM_IMAGE" \
        python /workspace/simulation/router_v2.py \
            --prefiller-hosts localhost \
            --prefiller-ports $prefill_ports \
            --decoder-hosts localhost \
            --decoder-ports $decode_ports \
            --port $router_port \
            --policy po2 \
            $mpc_flag
    
    # Wait for router
    sleep 5
    if curl -s "http://localhost:$router_port/health" | grep -q "ok"; then
        log "Router is ready"
        return 0
    else
        log "WARNING: Router may not be ready"
        return 0
    fi
}

# =============================================================================
# Run Benchmark
# =============================================================================

run_benchmark() {
    local router_port=$1
    local output_dir=$2
    local name=$3
    local duration=${4:-60}
    local rps=${5:-10}
    
    log "Running benchmark: $name (duration=${duration}s, RPS=$rps)"
    
    mkdir -p "$output_dir"
    
    # Run benchmark
    python "$PROJECT_ROOT/simulation/benchmark_v2.py" \
        --url "http://localhost:$router_port/v1/chat/completions" \
        --duration $duration \
        --rps $rps \
        --concurrency 32 \
        --output-dir "$output_dir" \
        --name "$name" \
        2>&1 | tee "$output_dir/$name.log"
    
    log "Benchmark $name completed"
}

# =============================================================================
# Main Experiment Loop
# =============================================================================

main() {
    mkdir -p "$RESULTS_DIR"
    log "Results will be saved to: $RESULTS_DIR"
    
    # Filter models/configs if specified
    local models_to_test=("${MODELS[@]}")
    if [ -n "$SELECTED_MODEL" ]; then
        models_to_test=("$SELECTED_MODEL")
    fi
    
    local configs_to_test=("${CONFIGS[@]}")
    if [ -n "$SELECTED_CONFIG" ]; then
        configs_to_test=("$SELECTED_CONFIG")
    fi
    
    # Quick mode: fewer experiments
    if [ "$QUICK_MODE" = "true" ]; then
        models_to_test=("${models_to_test[0]}")
        configs_to_test=("${configs_to_test[0]}")
    fi
    
    for model in "${models_to_test[@]}"; do
        log "===== Testing model: $model ====="
        
        for config in "${configs_to_test[@]}"; do
            IFS=':' read -r config_name prefill_tp decode_tp <<< "$config"
            log "=== Configuration: $config_name (P_TP=$prefill_tp, D_TP=$decode_tp) ==="
            
            cleanup
            
            # Port assignments
            PREFILL_PORT=8100
            DECODE_PORT=8200
            ROUTER_PORT=8000
            
            # Calculate GPU assignments
            PREFILL_GPU_START=0
            DECODE_GPU_START=$prefill_tp
            
            # Start servers
            start_prefill_server "p1" "$model" $prefill_tp $PREFILL_PORT $PREFILL_GPU_START || continue
            start_decode_server "d1" "$model" $decode_tp $DECODE_PORT $DECODE_GPU_START || continue
            
            # Test without MPC
            log "--- Testing WITHOUT MPC ---"
            start_orbit_router "$PREFILL_PORT" "$DECODE_PORT" $ROUTER_PORT "false"
            sleep 10
            run_benchmark $ROUTER_PORT "$RESULTS_DIR/${config_name}_baseline" "baseline" 60 10
            docker rm -f orbit_router 2>/dev/null
            
            # Test with MPC
            log "--- Testing WITH MPC ---"
            start_orbit_router "$PREFILL_PORT" "$DECODE_PORT" $ROUTER_PORT "true"
            sleep 10
            run_benchmark $ROUTER_PORT "$RESULTS_DIR/${config_name}_mpc" "mpc" 60 10
            
            cleanup
        done
    done
    
    log "===== All experiments completed ====="
    log "Results saved to: $RESULTS_DIR"
    
    # Generate comparison
    python "$PROJECT_ROOT/cluster_experiments/analysis/analyze_results.py" \
        "$RESULTS_DIR" \
        --output-dir "$RESULTS_DIR/figures" \
        2>&1 || log "Analysis script not available"
}

main "$@"
