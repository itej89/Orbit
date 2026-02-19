#!/bin/bash
# =============================================================================
# Orbit MPC Disaggregated vLLM Experiments
# Uses NixlConnector KV transfer with proper etcd coordination
# =============================================================================

set -e

# Configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta}"
PARTITION="${PARTITION:-amd-rccl}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/orbit_paper/cluster_experiments/results/$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Models to test (use non-gated models)
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
)

# Disaggregated configurations: name:xP:yD (xP=prefill nodes, yD=decode nodes)
# Each node uses all 8 GPUs with TP=8
CONFIGS=(
    "1p1d:1:1"     # 1 prefill + 1 decode = 2 nodes (+ 1 proxy = 3 total)
    "2p2d:2:2"     # 2 prefill + 2 decode = 4 nodes (+ 1 proxy = 5 total)
)

# Workload patterns
WORKLOADS=(
    "uniform_short:100:500:20:50"
    "variable_osl:200:400:10:500"
)

echo "=== Orbit MPC Disaggregated Experiments ==="
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
    local xP=$4
    local yD=$5
    local workload_name=$6
    local isl_min=$7
    local isl_max=$8
    local osl_min=$9
    local osl_max=${10}
    local mpc_enabled=${11}
    
    local total_nodes=$((1 + xP + yD))  # proxy + prefill + decode
    
    local script_path="$RESULTS_DIR/scripts/${job_name}.sbatch"
    
    cat > "$script_path" << 'SBATCH_EOF'
#!/bin/bash
#SBATCH --job-name=JOB_NAME
#SBATCH --partition=PARTITION
#SBATCH --nodes=TOTAL_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=RESULTS_DIR/logs/JOB_NAME_%j.out
#SBATCH --error=RESULTS_DIR/logs/JOB_NAME_%j.err

set -x

# =============================================================================
# Environment Setup
# =============================================================================

export xP=XP_VALUE
export yD=YD_VALUE
export MODEL_PATH="MODEL_VALUE"
export MODEL_NAME="MODEL_SHORT"
export DOCKER_IMAGE="DOCKER_IMAGE_VALUE"
export WORK_DIR="RESULTS_DIR/WORK_DIR_NAME"
export MPC_ENABLED="MPC_ENABLED_VALUE"
export ISL_MIN=ISL_MIN_VALUE
export ISL_MAX=ISL_MAX_VALUE
export OSL_MIN=OSL_MIN_VALUE
export OSL_MAX=OSL_MAX_VALUE

mkdir -p "$WORK_DIR"

# Create run_logs directory on all nodes
srun --ntasks-per-node=1 bash -c "mkdir -p /run_logs/${SLURM_JOB_ID}"

# Get node IPs
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
IPADDRS=""
for node in $NODELIST; do
    ip=$(getent hosts $node | awk '{print $1}')
    if [ -n "$IPADDRS" ]; then
        IPADDRS="${IPADDRS},"
    fi
    IPADDRS="${IPADDRS}${ip}"
done
export IPADDRS
export MASTER_ADDR=$(echo $IPADDRS | cut -d',' -f1)

echo "=== Orbit MPC Disaggregated Experiment ==="
echo "Job: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
echo "IPs: ${IPADDRS}"
echo "Config: ${xP}P${yD}D"
echo "Model: ${MODEL_PATH}"
echo "MPC: ${MPC_ENABLED}"

# =============================================================================
# Pull Docker Image on All Nodes
# =============================================================================

srun --ntasks-per-node=1 bash -c "docker pull ${DOCKER_IMAGE}"

# =============================================================================
# Start Experiment with Disaggregated Serving
# =============================================================================

export NIXL_COOKBOOK_PATH=/shared_inference/ravgupta/vllm_disagg_testing/MAD-private-pr134/scripts/vllm_dissag

# Run the disaggregated server script on each node
srun --ntasks-per-node=1 --export=ALL bash -c '
    export NODE_RANK=$SLURM_PROCID
    
    # Create local log directory
    mkdir -p /tmp/run_logs/${SLURM_JOB_ID}
    
    docker run --rm \
        --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --ipc=host \
        --shm-size=64GB \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -e NODE_RANK=$NODE_RANK \
        -e MASTER_ADDR=$MASTER_ADDR \
        -e IPADDRS=$IPADDRS \
        -e xP=$xP \
        -e yD=$yD \
        -e MODEL_PATH=$MODEL_PATH \
        -e MODEL_NAME=$MODEL_NAME \
        -e SLURM_JOB_ID=$SLURM_JOB_ID \
        -e NIXL_COOKBOOK_PATH=/cookbook \
        -v /tmp/run_logs:/run_logs \
        -v ${NIXL_COOKBOOK_PATH}:/cookbook:ro \
        ${DOCKER_IMAGE} \
        bash /cookbook/vllm_disagg_server.sh
'

echo "=== Experiment Complete ==="

# Collect results from all nodes
srun --ntasks-per-node=1 bash -c "cp -r /tmp/run_logs/${SLURM_JOB_ID}/* ${WORK_DIR}/ 2>/dev/null || true"

SBATCH_EOF

    # Replace placeholders
    sed -i "s|JOB_NAME|${job_name}|g" "$script_path"
    sed -i "s|PARTITION|${PARTITION}|g" "$script_path"
    sed -i "s|TOTAL_NODES|${total_nodes}|g" "$script_path"
    sed -i "s|RESULTS_DIR|${RESULTS_DIR}|g" "$script_path"
    sed -i "s|XP_VALUE|${xP}|g" "$script_path"
    sed -i "s|YD_VALUE|${yD}|g" "$script_path"
    sed -i "s|MODEL_VALUE|${model}|g" "$script_path"
    sed -i "s|MODEL_SHORT|$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.-')|g" "$script_path"
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
# Generate and Submit Jobs
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

# Quick mode: just 1p1d with one workload
if [ "$QUICK_MODE" = "true" ]; then
    CONFIGS=("1p1d:1:1")
    WORKLOADS=("uniform_short:100:500:20:50")
fi

JOB_COUNT=0

for model in "${MODELS[@]}"; do
    model_short=$(basename "$model" | tr '[:upper:]' '[:lower:]' | tr -d '.-')
    
    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r config_name xP yD <<< "$config"
        
        for workload in "${WORKLOADS[@]}"; do
            IFS=':' read -r workload_name isl_min isl_max osl_min osl_max <<< "$workload"
            
            # Baseline (no MPC)
            job_name="${model_short}_${config_name}_${workload_name}"
            script_path=$(generate_sbatch_script "$job_name" "$model" "$config_name" "$xP" "$yD" \
                "$workload_name" "$isl_min" "$isl_max" "$osl_min" "$osl_max" "false")
            echo "Generated: $job_name"
            
            if [ "$DRY_RUN" != "true" ]; then
                result=$(sbatch "$script_path" 2>&1)
                echo "  Submitted: $result"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
            
            # With MPC
            job_name="${model_short}_${config_name}_${workload_name}_mpc"
            script_path=$(generate_sbatch_script "$job_name" "$model" "$config_name" "$xP" "$yD" \
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
