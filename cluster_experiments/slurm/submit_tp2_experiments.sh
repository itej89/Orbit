#!/bin/bash
# =============================================================================
# Orbit MPC Experiments - TP=2 Focus for MI300X
# Multiple vLLM services at TP=2 on single node
# =============================================================================

set -e

# Configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta}"
PARTITION="${PARTITION:-amd-rccl}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/orbit_paper/cluster_experiments/results/tp2_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Models - using TP=2 friendly models
MODELS=(
    "Qwen/Qwen2.5-14B-Instruct"
)

# TP=2 configurations on 8 GPUs (MI300X)
# Each service uses 2 GPUs, so we can have up to 4 services
# Format: name:service1_tp:service1_gpus:service2_tp:service2_gpus:...
CONFIGS=(
    # Homogeneous: 4 equal TP=2 services (8 GPUs)
    "homo_4x2:2:0,1:2:2,3:2:4,5:2:6,7"
    
    # Homogeneous: 2 equal TP=2 services (4 GPUs)
    "homo_2x2:2:0,1:2:2,3"
    
    # Heterogeneous: 3 services (1 slow + 2 fast via MPC weights)
    "het_3x2:2:0,1:2:2,3:2:4,5"
    
    # Heterogeneous: 2 TP=2 + 1 TP=1 (5 GPUs)
    "het_2x2_1x1:2:0,1:2:2,3:1:4"
)

# Workload patterns with variable ISL/OSL
WORKLOADS=(
    "uniform_short:100:500:20:100"
    "variable_osl:200:400:10:500"
    "high_load:100:300:50:200"
)

echo "=== Orbit MPC TP=2 Experiments ==="
echo "Results directory: $RESULTS_DIR"
echo "Docker image: $DOCKER_IMAGE"
mkdir -p "$RESULTS_DIR/scripts" "$RESULTS_DIR/logs"

# =============================================================================
# Generate Slurm Job Script
# =============================================================================

generate_sbatch_script() {
    local job_name=$1
    local model=$2
    local config_name=$3
    local config_spec=$4
    local workload_name=$5
    local isl_min=$6
    local isl_max=$7
    local osl_min=$8
    local osl_max=$9
    local mpc_enabled=${10}
    
    local script_path="$RESULTS_DIR/scripts/${job_name}.sbatch"
    
    cat > "$script_path" << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --output=RESULTS_DIR/logs/JOB_NAME_%j.out
#SBATCH --error=RESULTS_DIR/logs/JOB_NAME_%j.err
#SBATCH --exclude=useocpm2m-097-094,useocpm2m-097-024

set -x

# =============================================================================
# Configuration
# =============================================================================

export MODEL_PATH="MODEL_VALUE"
export MODEL_NAME="MODEL_SHORT"
export CONFIG_NAME="CONFIG_NAME_VALUE"
export CONFIG_SPEC="CONFIG_SPEC_VALUE"
export DOCKER_IMAGE="DOCKER_IMAGE_VALUE"
export WORK_DIR="RESULTS_DIR/WORK_DIR_NAME"
export MPC_ENABLED="MPC_ENABLED_VALUE"
export ISL_MIN=ISL_MIN_VALUE
export ISL_MAX=ISL_MAX_VALUE
export OSL_MIN=OSL_MIN_VALUE
export OSL_MAX=OSL_MAX_VALUE
export ORBIT_CODE_DIR="${HOME}/orbit_paper/simulation"

mkdir -p "$WORK_DIR"

echo "=== Orbit MPC TP=2 Experiment ==="
echo "Job: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Config: ${CONFIG_NAME}"
echo "Model: ${MODEL_PATH}"
echo "MPC: ${MPC_ENABLED}"
echo "Workload: ISL=${ISL_MIN}-${ISL_MAX}, OSL=${OSL_MIN}-${OSL_MAX}"

# =============================================================================
# Clean up any existing containers THOROUGHLY
# =============================================================================

echo "Cleaning up existing containers..."
docker ps -a --filter "name=vllm_" -q | xargs -r docker rm -f 2>/dev/null || true
docker ps -a --filter "name=orbit_" -q | xargs -r docker rm -f 2>/dev/null || true
sleep 5

# =============================================================================
# Pull Docker Image
# =============================================================================

echo "Pulling Docker image..."
docker pull ${DOCKER_IMAGE}

# =============================================================================
# Parse config and start vLLM services
# =============================================================================

# CONFIG_SPEC format: tp1:gpus1:tp2:gpus2:...
IFS=':' read -ra CONFIG_PARTS <<< "${CONFIG_SPEC}"

BACKEND_URLS=""
SERVICE_IDX=0
BASE_PORT=8100

# Process pairs of (tp, gpus)
for ((i=0; i<${#CONFIG_PARTS[@]}; i+=2)); do
    TP=${CONFIG_PARTS[$i]}
    GPUS=${CONFIG_PARTS[$((i+1))]}
    PORT=$((BASE_PORT + SERVICE_IDX))
    SERVICE_NAME="vllm_service_${SERVICE_IDX}"
    
    echo "Starting ${SERVICE_NAME}: TP=${TP}, GPUs=${GPUS}, Port=${PORT}"
    
    # Use VLLM_USE_V1=0 to avoid v1 engine issues
    docker run -d \
        --name ${SERVICE_NAME} \
        --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=64GB \
        -e HIP_VISIBLE_DEVICES="${GPUS}" \
        -e VLLM_LOGGING_LEVEL=WARNING \
        -e VLLM_USE_V1=0 \
        ${DOCKER_IMAGE} \
        python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --tensor-parallel-size ${TP} \
        --port ${PORT} \
        --disable-log-requests \
        --trust-remote-code \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.85
    
    if [ -n "$BACKEND_URLS" ]; then
        BACKEND_URLS="${BACKEND_URLS},"
    fi
    BACKEND_URLS="${BACKEND_URLS}http://localhost:${PORT}"
    
    SERVICE_IDX=$((SERVICE_IDX + 1))
    
    # Brief pause between container starts
    sleep 3
done

NUM_SERVICES=$SERVICE_IDX
echo "Started ${NUM_SERVICES} vLLM services"
echo "Backend URLs: ${BACKEND_URLS}"

# =============================================================================
# Wait for all services to be ready (longer timeout for model loading)
# =============================================================================

echo "Waiting for all services to be ready..."
for ((i=0; i<NUM_SERVICES; i++)); do
    PORT=$((BASE_PORT + i))
    echo "Waiting for service on port ${PORT}..."
    for attempt in $(seq 1 180); do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health 2>/dev/null | grep -q "200"; then
            echo "Service on port ${PORT} is ready"
            break
        fi
        if [ $attempt -eq 180 ]; then
            echo "ERROR: Service on port ${PORT} did not start"
            docker logs vllm_service_$i 2>&1 | tail -100
            exit 1
        fi
        sleep 5
    done
done

echo "All services ready!"

# =============================================================================
# Start Orbit Router
# =============================================================================

echo "Starting Orbit Router..."
ROUTER_CMD="python /opt/orbit/standard_router.py --host 0.0.0.0 --port 8000 --policy po2"

# Add backend URLs
IFS=',' read -ra URLS <<< "${BACKEND_URLS}"
for url in "${URLS[@]}"; do
    ROUTER_CMD="${ROUTER_CMD} --backend ${url}"
done

# Add MPC if enabled
if [ "${MPC_ENABLED}" = "true" ]; then
    ROUTER_CMD="${ROUTER_CMD} --enable-mpc"
fi

docker run -d \
    --name orbit_router \
    --network=host \
    -v ${ORBIT_CODE_DIR}:/opt/orbit:ro \
    ${DOCKER_IMAGE} \
    ${ROUTER_CMD}

# Wait for router
echo "Waiting for router..."
for attempt in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null | grep -q "200"; then
        echo "Router ready"
        break
    fi
    sleep 2
done

# =============================================================================
# Run Benchmark
# =============================================================================

echo "Running benchmark..."
docker run --rm \
    --name orbit_benchmark \
    --network=host \
    -v ${WORK_DIR}:/results \
    -v ${ORBIT_CODE_DIR}:/opt/orbit:ro \
    ${DOCKER_IMAGE} \
    python /opt/orbit/benchmark_v2.py \
    --url http://localhost:8000/v1/chat/completions \
    --metrics-url http://localhost:8000/metrics \
    --requests 300 \
    --concurrency 12 \
    --pattern steady \
    --min-prompt ${ISL_MIN} \
    --max-prompt ${ISL_MAX} \
    --min-output ${OSL_MIN} \
    --max-output ${OSL_MAX} \
    --output-dir /results \
    --name ${CONFIG_NAME}

# =============================================================================
# Collect Results
# =============================================================================

echo "Collecting results..."

# Save service logs
for ((i=0; i<NUM_SERVICES; i++)); do
    docker logs vllm_service_$i > ${WORK_DIR}/vllm_service_${i}.log 2>&1 || true
done

docker logs orbit_router > ${WORK_DIR}/router.log 2>&1 || true

# Get router metrics
curl -s http://localhost:8000/metrics > ${WORK_DIR}/router_metrics.json 2>/dev/null || true

# =============================================================================
# Cleanup
# =============================================================================

echo "Cleaning up..."
docker ps -a --filter "name=vllm_" -q | xargs -r docker rm -f 2>/dev/null || true
docker ps -a --filter "name=orbit_" -q | xargs -r docker rm -f 2>/dev/null || true

echo "=== Experiment Complete ==="
echo "Results saved to: ${WORK_DIR}"

SBATCH_EOF

    # Replace placeholders
    sed -i "s|JOB_NAME|${job_name}|g" "$script_path"
    sed -i "s|PARTITION|${PARTITION}|g" "$script_path"
    sed -i "s|RESULTS_DIR|${RESULTS_DIR}|g" "$script_path"
    sed -i "s|MODEL_VALUE|${model}|g" "$script_path"
    sed -i "s|MODEL_SHORT|$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.-')|g" "$script_path"
    sed -i "s|CONFIG_NAME_VALUE|${config_name}|g" "$script_path"
    sed -i "s|CONFIG_SPEC_VALUE|${config_spec}|g" "$script_path"
    sed -i "s|DOCKER_IMAGE_VALUE|${DOCKER_IMAGE}|g" "$script_path"
    sed -i "s|WORK_DIR_NAME|${job_name}|g" "$script_path"
    sed -i "s|MPC_ENABLED_VALUE|${mpc_enabled}|g" "$script_path"
    sed -i "s|ISL_MIN_VALUE|${isl_min}|g" "$script_path"
    sed -i "s|ISL_MAX_VALUE|${isl_max}|g" "$script_path"
    sed -i "s|OSL_MIN_VALUE|${osl_min}|g" "$script_path"
    sed -i "s|OSL_MAX_VALUE|${osl_max}|g" "$script_path"
    
    chmod +x "$script_path"
    echo "$script_path"
}

# =============================================================================
# Parse Arguments
# =============================================================================

QUICK_MODE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Quick mode: minimal test
if [ "$QUICK_MODE" = "true" ]; then
    CONFIGS=("homo_2x2:2:0,1:2:2,3")
    WORKLOADS=("uniform_short:100:500:20:100")
fi

# =============================================================================
# Generate and Submit Jobs
# =============================================================================

JOB_COUNT=0

for model in "${MODELS[@]}"; do
    model_short=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.-')
    
    for config in "${CONFIGS[@]}"; do
        # Split config into name and spec
        config_name="${config%%:*}"
        config_spec="${config#*:}"
        
        for workload in "${WORKLOADS[@]}"; do
            IFS=':' read -r workload_name isl_min isl_max osl_min osl_max <<< "$workload"
            
            # Baseline (no MPC)
            job_name="${model_short}_${config_name}_${workload_name}"
            script_path=$(generate_sbatch_script "$job_name" "$model" "$config_name" "$config_spec" \
                "$workload_name" "$isl_min" "$isl_max" "$osl_min" "$osl_max" "false")
            echo "Generated: $job_name"
            
            if [ "$DRY_RUN" != "true" ]; then
                result=$(sbatch "$script_path" 2>&1)
                echo "  Submitted: $result"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
            
            # With MPC
            job_name="${model_short}_${config_name}_${workload_name}_mpc"
            script_path=$(generate_sbatch_script "$job_name" "$model" "$config_name" "$config_spec" \
                "$workload_name" "$isl_min" "$isl_max" "$osl_min" "$osl_max" "true")
            echo "Generated: $job_name"
            
            if [ "$DRY_RUN" != "true" ]; then
                result=$(sbatch "$script_path" 2>&1)
                echo "  Submitted: $result"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total jobs: $JOB_COUNT"
echo "Scripts saved to: $RESULTS_DIR/scripts/"
echo ""
echo "Experiment Matrix:"
echo "  Models: ${MODELS[*]}"
echo "  Configs: ${#CONFIGS[@]} TP=2 configurations"
echo "  Workloads: ${#WORKLOADS[@]} patterns"
echo "  Each config tested with and without MPC"
