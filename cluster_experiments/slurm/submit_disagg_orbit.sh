#!/bin/bash
# =============================================================================
# Orbit MPC Disaggregated Experiments - Expanded
# Targets specific idle nodes, tests homo/hetero configurations
# =============================================================================

set -e

DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta}"
PARTITION="${PARTITION:-amd-rccl}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/orbit_paper/cluster_experiments/results/disagg_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Available idle nodes (8 total)
NODE_POOL="useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027,useocpm2m-097-042,useocpm2m-097-043,useocpm2m-097-047"

# =============================================================================
# Model
# =============================================================================

MODEL_NAME="Qwen14B"
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"

# =============================================================================
# Experiment Phases
# Phase 1: 2P2D homogeneous baseline (5 nodes)
# Phase 2: 2P2D heterogeneous - MPC showcase (5 nodes)
# Phase 3: 3P3D homogeneous scale-out (7 nodes)
# Phase 4: 3P3D heterogeneous full showcase (7 nodes)
# =============================================================================

# Format: name:num_prefill:num_decode:num_nodes:nodelist
# Node assignment:
#   2P2D: proxy=019, P=020+023, D=025+027 (5 nodes)
#   3P3D: proxy=019, P=020+023+025, D=027+042+043 (7 nodes)
CONFIGS_2P2D="2P2D:2:2:5:useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027"
CONFIGS_3P3D="3P3D:3:3:7:useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027,useocpm2m-097-042,useocpm2m-097-043"

echo "=== Orbit MPC Disaggregated Experiments ==="
echo "Results: $RESULTS_DIR"
echo "Image: $DOCKER_IMAGE"
mkdir -p "$RESULTS_DIR/scripts" "$RESULTS_DIR/logs"

# =============================================================================
# Generate Slurm Script
# =============================================================================

generate_sbatch_script() {
    local job_name=$1
    local num_prefill=$2
    local num_decode=$3
    local total_nodes=$4
    local nodelist=$5
    local mpc_enabled=$6
    local bench_isl=$7
    local bench_osl=$8
    local bench_concurrency=$9
    local bench_prompts=${10}

    local script_path="$RESULTS_DIR/scripts/${job_name}.sbatch"

    cat > "$script_path" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${total_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=${RESULTS_DIR}/logs/${job_name}_%j.out
#SBATCH --error=${RESULTS_DIR}/logs/${job_name}_%j.err
#SBATCH --exclusive
#SBATCH --nodelist=${nodelist}

set -x

export MODEL_NAME="${MODEL_NAME}"
export MODEL_PATH="${MODEL_PATH}"
export DOCKER_IMAGE="${DOCKER_IMAGE}"
export WORK_DIR="${RESULTS_DIR}/${job_name}"
export MPC_ENABLED="${mpc_enabled}"
export xP=${num_prefill}
export yD=${num_decode}
export ORBIT_CODE_DIR="\${HOME}/orbit_paper/simulation"
export NIXL_COOKBOOK_PATH="\${HOME}/orbit_paper/cluster_experiments/slurm"
export BENCH_ISL="${bench_isl}"
export BENCH_OSL="${bench_osl}"
export BENCH_CONCURRENCY="${bench_concurrency}"
export BENCH_PROMPTS="${bench_prompts}"

mkdir -p "\$WORK_DIR"
mkdir -p /tmp/run_logs/\${SLURM_JOB_ID}

echo "=== Orbit Disaggregated Experiment ==="
echo "Job: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NODELIST}"
echo "Config: ${num_prefill}P / ${num_decode}D"
echo "Model: \${MODEL_PATH}"
echo "MPC: \${MPC_ENABLED}"
echo "Benchmark: ISL=\${BENCH_ISL} OSL=\${BENCH_OSL} CON=\${BENCH_CONCURRENCY} PROMPTS=\${BENCH_PROMPTS}"

# Get Node IPs
NODELIST=\$(scontrol show hostname \$SLURM_JOB_NODELIST)
IPADDRS=""
for node in \$NODELIST; do
    ip=\$(getent hosts \$node | awk '{print \$1}')
    if [ -n "\$IPADDRS" ]; then
        IPADDRS="\${IPADDRS},"
    fi
    IPADDRS="\${IPADDRS}\${ip}"
done

MASTER_ADDR=\$(echo \$IPADDRS | cut -d',' -f1)
echo "Node IPs: \$IPADDRS"
echo "Master: \$MASTER_ADDR"

export IPADDRS
export MASTER_ADDR

# Pull Docker Image
echo "Pulling Docker image on all nodes..."
srun --ntasks-per-node=1 bash -c "docker pull \${DOCKER_IMAGE}" || true

echo "Creating run_logs directory on all nodes..."
srun --ntasks-per-node=1 bash -c "mkdir -p /tmp/run_logs/\${SLURM_JOB_ID}"

echo "Launching disaggregated servers..."

srun --ntasks-per-node=1 --export=ALL bash -c '
    NODE_RANK=\$SLURM_PROCID
    HOST_NIXL_PATH="\${NIXL_COOKBOOK_PATH}"
    HOST_ORBIT_PATH="\${ORBIT_CODE_DIR}"

    PORT_CONFIG_HOST="/shared_inference/ravgupta/orbit_ports"
    mkdir -p \${PORT_CONFIG_HOST} 2>/dev/null || true

    docker run --rm \
        --name orbit_node_\${NODE_RANK}_\${SLURM_JOB_ID} \
        --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --device=/dev/infiniband \
        --privileged \
        --group-add video \
        --ipc=host \
        --shm-size=128GB \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -e MODEL_PATH="\${MODEL_PATH}" \
        -e MODEL_NAME="\${MODEL_NAME}" \
        -e xP="\${xP}" \
        -e yD="\${yD}" \
        -e IPADDRS="\${IPADDRS}" \
        -e MASTER_ADDR="\${MASTER_ADDR}" \
        -e NODE_RANK="\${NODE_RANK}" \
        -e SLURM_JOB_ID="\${SLURM_JOB_ID}" \
        -e MPC_ENABLED="\${MPC_ENABLED}" \
        -e BENCH_ISL="\${BENCH_ISL}" \
        -e BENCH_OSL="\${BENCH_OSL}" \
        -e BENCH_CONCURRENCY="\${BENCH_CONCURRENCY}" \
        -e BENCH_PROMPTS="\${BENCH_PROMPTS}" \
        -e NIXL_COOKBOOK_PATH=/app/nixl_cookbook \
        -e PORT_CONFIG_DIR=/port_config \
        -v /tmp/run_logs:/run_logs \
        -v \${PORT_CONFIG_HOST}:/port_config \
        -v /shared_inference:/shared_inference \
        -v \${HOST_ORBIT_PATH}:/opt/orbit:ro \
        -v \${HOST_NIXL_PATH}:/app/nixl_cookbook:ro \
        \${DOCKER_IMAGE} \
        bash /app/nixl_cookbook/vllm_disagg_server_orbit.sh
'

# Collect Results
echo "Collecting results..."
cp -r /tmp/run_logs/\${SLURM_JOB_ID}/* \${WORK_DIR}/ 2>/dev/null || true

echo "=== Experiment Complete ==="
echo "Results: \${WORK_DIR}"
SBATCH_EOF

    chmod +x "$script_path"
    echo "$script_path"
}

# =============================================================================
# Parse Arguments
# =============================================================================

PHASE=""
DRY_RUN=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --quick) QUICK_MODE=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# =============================================================================
# Submit Experiments by Phase
# =============================================================================

JOB_COUNT=0

submit_pair() {
    local config_spec=$1
    local bench_isl=$2
    local bench_osl=$3
    local bench_con=$4
    local bench_prompts=$5
    local label=$6

    IFS=':' read -r config_name num_prefill num_decode total_nodes nodelist <<< "$config_spec"

    # Baseline (no MPC)
    local job_name="${MODEL_NAME}_${config_name}_${label}"
    local script_path=$(generate_sbatch_script \
        "$job_name" "$num_prefill" "$num_decode" "$total_nodes" "$nodelist" \
        "false" "$bench_isl" "$bench_osl" "$bench_con" "$bench_prompts")
    echo "Generated: $job_name (${num_prefill}P/${num_decode}D, ${total_nodes} nodes)"

    if [ "$DRY_RUN" != "true" ]; then
        result=$(sbatch "$script_path" 2>&1)
        echo "  $result"
        sleep 2
    fi
    JOB_COUNT=$((JOB_COUNT + 1))

    # With MPC
    job_name="${MODEL_NAME}_${config_name}_${label}_mpc"
    script_path=$(generate_sbatch_script \
        "$job_name" "$num_prefill" "$num_decode" "$total_nodes" "$nodelist" \
        "true" "$bench_isl" "$bench_osl" "$bench_con" "$bench_prompts")
    echo "Generated: $job_name (MPC enabled)"

    if [ "$DRY_RUN" != "true" ]; then
        result=$(sbatch "$script_path" 2>&1)
        echo "  $result"
        sleep 2
    fi
    JOB_COUNT=$((JOB_COUNT + 1))
}

if [ "$QUICK_MODE" = "true" ]; then
    echo "=== Quick Mode: 2P2D uniform baseline ==="
    submit_pair "$CONFIGS_2P2D" "128" "128" "4" "32" "quick"
    PHASE="done"
fi

if [ "$PHASE" = "1" ] || [ -z "$PHASE" ]; then
    echo ""
    echo "=== Phase 1: 2P2D Homogeneous - Multiple Workloads ==="
    echo "  Tests load balancing benefit of MPC with identical servers"
    echo ""

    # Low load - expect minimal MPC benefit (no imbalance to correct)
    submit_pair "$CONFIGS_2P2D" "128" "128" "4" "32" "low_load"

    # Medium load - start seeing routing benefit
    submit_pair "$CONFIGS_2P2D" "256" "256" "8" "64" "med_load"

    # High concurrency - MPC should shine with queue management
    submit_pair "$CONFIGS_2P2D" "256" "256" "16" "128" "high_con"

    # Variable ISL/OSL - creates processing time imbalance
    submit_pair "$CONFIGS_2P2D" "64" "512" "8" "64" "var_isl_osl"

fi

if [ "$PHASE" = "2" ] || ([ -z "$PHASE" ] && [ "$QUICK_MODE" != "true" ]); then
    echo ""
    echo "=== Phase 2: 3P3D Scale-Out ==="
    echo "  Tests if MPC benefit holds at larger scale"
    echo ""

    submit_pair "$CONFIGS_3P3D" "256" "256" "8" "64" "med_load"

    submit_pair "$CONFIGS_3P3D" "256" "256" "16" "128" "high_con"

    submit_pair "$CONFIGS_3P3D" "64" "512" "8" "64" "var_isl_osl"
fi

echo ""
echo "=== Summary ==="
echo "Total jobs: $JOB_COUNT"
echo "Scripts: $RESULTS_DIR/scripts/"
echo ""
echo "Node Assignments:"
echo "  2P2D: proxy=019, P=020+023, D=025+027"
echo "  3P3D: proxy=019, P=020+023+025, D=027+042+043"
