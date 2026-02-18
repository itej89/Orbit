#!/bin/bash
#
# Orbit MPC Experiment Submission Script
#
# Submits a comprehensive set of experiments to test MPC-augmented routing
# with heterogeneous prefill/decode server configurations.
#
# Usage:
#   ./submit_orbit_experiments.sh [--model MODEL] [--quick] [--dry-run]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# =============================================================================
# Configuration
# =============================================================================

PARTITION="${PARTITION:-amd-rccl}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_orbit_mpc}"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_DIR/cluster_experiments/results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$RESULTS_BASE/$TIMESTAMP"

# Models to test
MODELS=(
    "meta-llama/Llama-3.1-70B-Instruct"
    # "Qwen/Qwen1.5-MoE-A2.7B"
    # "allenai/OLMoE-1B-7B"
)

# Server configurations: name:prefill_tp:decode_tp:prefill_gpus:decode_gpus
# Format allows testing heterogeneous TP configurations
CONFIGS=(
    # Homogeneous configs
    "homo_8x8:8:8:8:8"          # 8 GPU prefill + 8 GPU decode (16 GPUs total)
    
    # Heterogeneous configs (different TP)
    "het_8x4:8:4:8:4"           # 8 GPU prefill + 4 GPU decode (12 GPUs)
    "het_8x2:8:2:8:2"           # 8 GPU prefill + 2 GPU decode (10 GPUs)
    "het_4x2:4:2:4:2"           # 4 GPU prefill + 2 GPU decode (6 GPUs)
    
    # Multi-instance heterogeneous (2 prefill + 2 decode)
    "het_2p2d_4x2:4:2:8:4"      # 2x4GPU prefill + 2x2GPU decode (12 GPUs)
)

# Workload patterns with variable ISL/OSL
WORKLOADS=(
    "uniform_short:100:500:20:50"        # ISL 100-500, OSL 20-50 (short responses)
    "uniform_long:100:500:100:500"       # ISL 100-500, OSL 100-500 (long responses)
    "variable_isl:50:2000:50:100"        # Variable ISL 50-2000, short OSL
    "variable_osl:200:400:10:500"        # Fixed ISL, variable OSL 10-500
    "high_variance:50:3000:10:1000"      # High variance both
)

# =============================================================================
# Parse Arguments
# =============================================================================

QUICK_MODE=false
DRY_RUN=false
SELECTED_MODEL=""
SELECTED_CONFIG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) SELECTED_MODEL="$2"; shift 2 ;;
        --config) SELECTED_CONFIG="$2"; shift 2 ;;
        --quick) QUICK_MODE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Functions
# =============================================================================

generate_sbatch_script() {
    local job_name=$1
    local model=$2
    local config_name=$3
    local prefill_tp=$4
    local decode_tp=$5
    local prefill_gpus=$6
    local decode_gpus=$7
    local workload_name=$8
    local isl_min=$9
    local isl_max=${10}
    local osl_min=${11}
    local osl_max=${12}
    local mpc_enabled=${13}
    
    local total_gpus=$((prefill_gpus + decode_gpus))
    local nodes_needed=$(( (total_gpus + 7) / 8 ))
    
    local script_path="$RESULTS_DIR/scripts/${job_name}.sbatch"
    mkdir -p "$(dirname "$script_path")"
    
    cat > "$script_path" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${nodes_needed}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=${RESULTS_DIR}/logs/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/logs/${job_name}_%j.err

set -e

echo "=== Orbit MPC Experiment ==="
echo "Job: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Config: ${config_name}"
echo "Model: ${model}"
echo "Workload: ${workload_name}"
echo "MPC: ${mpc_enabled}"
echo "ISL: ${isl_min}-${isl_max}, OSL: ${osl_min}-${osl_max}"
echo ""

# Pull Docker image if needed
docker pull ${DOCKER_IMAGE}

# Setup directories
WORK_DIR="${RESULTS_DIR}/${job_name}"
mkdir -p "\$WORK_DIR"

# Kill any existing containers
docker ps -a --filter "name=orbit_" -q | xargs -r docker rm -f 2>/dev/null || true

# =============================================================================
# Start Prefill Server(s)
# =============================================================================

echo "Starting prefill server (TP=${prefill_tp})..."

# Calculate GPU indices for prefill
PREFILL_GPUS=""
for i in \$(seq 0 $((prefill_gpus - 1))); do
    if [ -n "\$PREFILL_GPUS" ]; then
        PREFILL_GPUS="\${PREFILL_GPUS},"
    fi
    PREFILL_GPUS="\${PREFILL_GPUS}\$i"
done

docker run -d \\
    --name orbit_prefill \\
    --network=host \\
    --device=/dev/kfd --device=/dev/dri \\
    --group-add video \\
    --ipc=host \\
    --shm-size=64GB \\
    -e HIP_VISIBLE_DEVICES="\$PREFILL_GPUS" \\
    -e VLLM_LOGGING_LEVEL=INFO \\
    ${DOCKER_IMAGE} \\
    vllm --model "${model}" \\
    --tensor-parallel-size ${prefill_tp} \\
    --port 8100 \\
    --kv-transfer-config '{"kv_connector": "PyNcclConnector"}' \\
    --no-enable-prefix-caching

# Wait for prefill server
echo "Waiting for prefill server..."
for i in \$(seq 1 120); do
    if curl -s http://localhost:8100/health | grep -q "ok"; then
        echo "Prefill server ready"
        break
    fi
    sleep 5
done

# =============================================================================
# Start Decode Server(s)
# =============================================================================

echo "Starting decode server (TP=${decode_tp})..."

# Calculate GPU indices for decode (after prefill)
DECODE_GPUS=""
for i in \$(seq ${prefill_gpus} $((prefill_gpus + decode_gpus - 1))); do
    if [ -n "\$DECODE_GPUS" ]; then
        DECODE_GPUS="\${DECODE_GPUS},"
    fi
    DECODE_GPUS="\${DECODE_GPUS}\$i"
done

docker run -d \\
    --name orbit_decode \\
    --network=host \\
    --device=/dev/kfd --device=/dev/dri \\
    --group-add video \\
    --ipc=host \\
    --shm-size=64GB \\
    -e HIP_VISIBLE_DEVICES="\$DECODE_GPUS" \\
    -e VLLM_LOGGING_LEVEL=INFO \\
    ${DOCKER_IMAGE} \\
    vllm --model "${model}" \\
    --tensor-parallel-size ${decode_tp} \\
    --port 8200 \\
    --kv-transfer-config '{"kv_connector": "PyNcclConnector"}'

# Wait for decode server
echo "Waiting for decode server..."
for i in \$(seq 1 120); do
    if curl -s http://localhost:8200/health | grep -q "ok"; then
        echo "Decode server ready"
        break
    fi
    sleep 5
done

# =============================================================================
# Start MPC Controller (if enabled)
# =============================================================================

if [ "${mpc_enabled}" = "true" ]; then
    echo "Starting MPC Controller..."
    docker run -d \\
        --name orbit_mpc \\
        --network=host \\
        ${DOCKER_IMAGE} \\
        mpc-controller --port 8090
    
    sleep 5
fi

# =============================================================================
# Start Orbit Router
# =============================================================================

echo "Starting Orbit Router..."

MPC_FLAGS=""
if [ "${mpc_enabled}" = "true" ]; then
    MPC_FLAGS="--enable-mpc --mpc-horizon 10"
fi

docker run -d \\
    --name orbit_router \\
    --network=host \\
    ${DOCKER_IMAGE} \\
    router \\
    --prefiller-hosts localhost \\
    --prefiller-ports 8100 \\
    --decoder-hosts localhost \\
    --decoder-ports 8200 \\
    --port 8000 \\
    --policy po2 \\
    \$MPC_FLAGS

sleep 10

# =============================================================================
# Run Benchmark
# =============================================================================

echo "Running benchmark..."

docker run \\
    --name orbit_benchmark \\
    --network=host \\
    -v "\$WORK_DIR:/results" \\
    ${DOCKER_IMAGE} \\
    benchmark \\
    --url http://localhost:8000/v1/chat/completions \\
    --duration 120 \\
    --rps 10 \\
    --concurrency 32 \\
    --isl-min ${isl_min} \\
    --isl-max ${isl_max} \\
    --osl-min ${osl_min} \\
    --osl-max ${osl_max} \\
    --output-dir /results \\
    --name "${job_name}"

# =============================================================================
# Collect Results
# =============================================================================

echo "Collecting results..."

# Get router metrics
curl -s http://localhost:8000/metrics > "\$WORK_DIR/router_metrics.json" 2>/dev/null || true

# Get container logs
docker logs orbit_prefill > "\$WORK_DIR/prefill.log" 2>&1 || true
docker logs orbit_decode > "\$WORK_DIR/decode.log" 2>&1 || true
docker logs orbit_router > "\$WORK_DIR/router.log" 2>&1 || true
if [ "${mpc_enabled}" = "true" ]; then
    docker logs orbit_mpc > "\$WORK_DIR/mpc.log" 2>&1 || true
fi

# Cleanup
docker rm -f orbit_prefill orbit_decode orbit_router orbit_mpc orbit_benchmark 2>/dev/null || true

echo "Experiment complete: \$WORK_DIR"
SBATCH_EOF

    chmod +x "$script_path"
    echo "$script_path"
}

# =============================================================================
# Main
# =============================================================================

mkdir -p "$RESULTS_DIR/scripts" "$RESULTS_DIR/logs"

echo "=== Orbit MPC Cluster Experiments ==="
echo "Results directory: $RESULTS_DIR"
echo "Partition: $PARTITION"
echo "Docker image: $DOCKER_IMAGE"
echo ""

# Filter configs/models if specified
models_to_test=("${MODELS[@]}")
if [ -n "$SELECTED_MODEL" ]; then
    models_to_test=("$SELECTED_MODEL")
fi

configs_to_test=("${CONFIGS[@]}")
if [ -n "$SELECTED_CONFIG" ]; then
    configs_to_test=("$SELECTED_CONFIG")
fi

# Quick mode: fewer experiments
if [ "$QUICK_MODE" = "true" ]; then
    models_to_test=("${models_to_test[0]}")
    configs_to_test=("homo_8x8:8:8:8:8" "het_8x4:8:4:8:4")
    WORKLOADS=("uniform_short:100:500:20:50" "variable_osl:200:400:10:500")
fi

JOB_COUNT=0

for model in "${models_to_test[@]}"; do
    model_short=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.-')
    
    for config in "${configs_to_test[@]}"; do
        IFS=':' read -r config_name prefill_tp decode_tp prefill_gpus decode_gpus <<< "$config"
        
        for workload in "${WORKLOADS[@]}"; do
            IFS=':' read -r workload_name isl_min isl_max osl_min osl_max <<< "$workload"
            
            for mpc_enabled in "false" "true"; do
                mpc_suffix=""
                if [ "$mpc_enabled" = "true" ]; then
                    mpc_suffix="_mpc"
                fi
                
                job_name="${model_short}_${config_name}_${workload_name}${mpc_suffix}"
                
                script=$(generate_sbatch_script \
                    "$job_name" "$model" "$config_name" \
                    "$prefill_tp" "$decode_tp" "$prefill_gpus" "$decode_gpus" \
                    "$workload_name" "$isl_min" "$isl_max" "$osl_min" "$osl_max" \
                    "$mpc_enabled")
                
                echo "Generated: $job_name"
                
                if [ "$DRY_RUN" = "false" ]; then
                    sbatch_out=$(sbatch "$script" 2>&1)
                    echo "  Submitted: $sbatch_out"
                fi
                
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total jobs: $JOB_COUNT"
echo "Scripts saved to: $RESULTS_DIR/scripts/"

if [ "$DRY_RUN" = "true" ]; then
    echo ""
    echo "DRY RUN - Jobs not submitted. Use without --dry-run to submit."
fi
