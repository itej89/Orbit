#!/bin/bash
# =============================================================================
# Orbit MPC - Multi-Model Comprehensive Experiment Suite
# =============================================================================
#
# Runs the full 5-phase experiment matrix for multiple models.
# Each model gets 40 experiments (same phases as Qwen2.5-14B-Instruct).
#
# Usage:
#   ./submit_multimodel_experiments.sh --model gpt-oss-120b --phase all
#   ./submit_multimodel_experiments.sh --model all --phase all
#   ./submit_multimodel_experiments.sh --model all --phase all --dry-run
#   ./submit_multimodel_experiments.sh --model all --phase all --sequential
# =============================================================================

set -e

DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta}"
PARTITION="${PARTITION:-amd-rccl}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default node pools (can be overridden with --nodes-2p2d and --nodes-3p3d)
NODE_POOL="${NODES_POOL:-useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027,useocpm2m-097-042,useocpm2m-097-043,useocpm2m-097-047}"
NODES_2P2D="${NODES_2P2D:-useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027}"
NODES_3P3D="${NODES_3P3D:-useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027,useocpm2m-097-042,useocpm2m-097-043}"

MODELS_BASE_DIR="/shared_inference/ravgupta/models"

declare -A MODEL_NAMES=(
    ["gpt-oss-120b"]="gpt-oss-120b"
    ["gpt-oss-20b"]="gpt-oss-20b"
    ["Qwen3-30B-A3B"]="Qwen3-30B-A3B"
    ["DBRX-Instruct"]="DBRX-Instruct"
    ["GLM-4.7-Flash"]="GLM-4.7-Flash"
)

declare -A MODEL_PATHS=(
    ["gpt-oss-120b"]="${MODELS_BASE_DIR}/gpt-oss-120b"
    ["gpt-oss-20b"]="${MODELS_BASE_DIR}/gpt-oss-20b"
    ["Qwen3-30B-A3B"]="${MODELS_BASE_DIR}/Qwen3-30B-A3B"
    ["DBRX-Instruct"]="${MODELS_BASE_DIR}/DBRX-Instruct"
    ["GLM-4.7-Flash"]="${MODELS_BASE_DIR}/GLM-4.7-Flash"
)

declare -A MODEL_SHORT=(
    ["gpt-oss-120b"]="gptoss120b"
    ["gpt-oss-20b"]="gptoss20b"
    ["Qwen3-30B-A3B"]="qwen3_30b"
    ["DBRX-Instruct"]="dbrx"
    ["GLM-4.7-Flash"]="glm47flash"
)

ALL_MODELS="gpt-oss-120b gpt-oss-20b Qwen3-30B-A3B GLM-4.7-Flash"

# =============================================================================
# Argument Parsing
# =============================================================================

TARGET_MODEL="all"
PHASE="all"
DRY_RUN=false
SEQUENTIAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) TARGET_MODEL="$2"; shift 2 ;;
        --phase) PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --sequential) SEQUENTIAL=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --nodes-2p2d) NODES_2P2D="$2"; shift 2 ;;
        --nodes-3p3d) NODES_3P3D="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ "$TARGET_MODEL" == "all" ]]; then
    MODELS_TO_RUN="$ALL_MODELS"
else
    MODELS_TO_RUN="$TARGET_MODEL"
fi

echo "=============================================="
echo "  Orbit MPC Multi-Model Experiments"
echo "=============================================="
echo "Models:     $MODELS_TO_RUN"
echo "Phase:      $PHASE"
echo "Dry run:    $DRY_RUN"
echo "Sequential: $SEQUENTIAL"
echo "=============================================="

JOB_COUNT=0
SUBMITTED_JOBS=()

# =============================================================================
# Job Generation (with /datasets mount for local model paths)
# =============================================================================

generate_job() {
    local job_name=$1
    local num_prefill=$2
    local num_decode=$3
    local total_nodes=$4
    local nodelist=$5
    local proxy_policy=$6
    local bench_isl=$7
    local bench_osl=$8
    local bench_concurrency=$9
    local bench_prompts=${10}
    local hetero_mode=${11:-false}
    local hetero_gpu_mem=${12:-0.45}
    local hetero_max_seqs=${13:-64}
    local model_name=${14}
    local model_path=${15}
    local results_dir=${16}

    local script_path="${results_dir}/scripts/${job_name}.sbatch"

    cat > "$script_path" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${total_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --output=${results_dir}/logs/${job_name}_%j.out
#SBATCH --error=${results_dir}/logs/${job_name}_%j.err
#SBATCH --exclusive
#SBATCH --nodelist=${nodelist}

set -x

export MODEL_NAME="${model_name}"
export MODEL_PATH="${model_path}"
export DOCKER_IMAGE="${DOCKER_IMAGE}"
export WORK_DIR="${results_dir}/${job_name}"
export PROXY_POLICY="${proxy_policy}"
export xP=${num_prefill}
export yD=${num_decode}
export ORBIT_CODE_DIR="\${HOME}/orbit_paper/simulation"
export NIXL_COOKBOOK_PATH="\${HOME}/orbit_paper/cluster_experiments/slurm"
export BENCH_ISL="${bench_isl}"
export BENCH_OSL="${bench_osl}"
export BENCH_CONCURRENCY="${bench_concurrency}"
export BENCH_PROMPTS="${bench_prompts}"
export HETERO_MODE="${hetero_mode}"
export HETERO_GPU_MEM="${hetero_gpu_mem}"
export HETERO_MAX_SEQS="${hetero_max_seqs}"

mkdir -p "\$WORK_DIR"

# Use per-job temp directory to avoid permission issues on shared /tmp/run_logs
export RUN_LOGS_DIR="/tmp/orbit_logs_\${SLURM_JOB_ID}"

echo "=========================================="
echo "  Orbit Experiment: ${job_name}"
echo "  Model: ${model_name} (${model_path})"
echo "=========================================="
echo "Job: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NODELIST}"
echo "Config: ${num_prefill}P / ${num_decode}D"
echo "Policy: ${proxy_policy}"
echo "Hetero: ${hetero_mode}"
echo "Benchmark: ISL=${bench_isl} OSL=${bench_osl} CON=${bench_concurrency} PROMPTS=${bench_prompts}"

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
echo "IPs: \$IPADDRS | Master: \$MASTER_ADDR"

export IPADDRS MASTER_ADDR

echo "Pulling image and preparing log dirs..."
srun --ntasks-per-node=1 bash -c "docker pull \${DOCKER_IMAGE}" || true
srun --ntasks-per-node=1 bash -c "mkdir -p \${RUN_LOGS_DIR} 2>/dev/null || mkdir -p /tmp/run_logs/\${SLURM_JOB_ID}"

echo "Launching servers..."
srun --ntasks-per-node=1 --export=ALL bash -c '
    NODE_RANK=\$SLURM_PROCID
    HOST_NIXL_PATH="\${NIXL_COOKBOOK_PATH}"
    HOST_ORBIT_PATH="\${ORBIT_CODE_DIR}"

    PORT_CONFIG_HOST="/shared_inference/ravgupta/orbit_ports"
    mkdir -p \${PORT_CONFIG_HOST} 2>/dev/null || true

    # Kill any leftover containers from previous jobs
    docker rm -f orbit_\${SLURM_JOB_ID}_\${NODE_RANK} 2>/dev/null || true

    docker run --rm \
        --name orbit_\${SLURM_JOB_ID}_\${NODE_RANK} \
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
        -e PROXY_POLICY="\${PROXY_POLICY}" \
        -e BENCH_ISL="\${BENCH_ISL}" \
        -e BENCH_OSL="\${BENCH_OSL}" \
        -e BENCH_CONCURRENCY="\${BENCH_CONCURRENCY}" \
        -e BENCH_PROMPTS="\${BENCH_PROMPTS}" \
        -e HETERO_MODE="\${HETERO_MODE}" \
        -e HETERO_GPU_MEM="\${HETERO_GPU_MEM}" \
        -e HETERO_MAX_SEQS="\${HETERO_MAX_SEQS}" \
        -e NIXL_COOKBOOK_PATH=/app/nixl_cookbook \
        -e PORT_CONFIG_DIR=/port_config \
        -v \${RUN_LOGS_DIR}:/run_logs \
        -v \${PORT_CONFIG_HOST}:/port_config \
        -v /shared_inference:/shared_inference \
        -v /datasets:/datasets \
        -v \${HOST_ORBIT_PATH}:/opt/orbit:ro \
        -v \${HOST_NIXL_PATH}:/app/nixl_cookbook:ro \
        \${DOCKER_IMAGE} \
        bash /app/nixl_cookbook/vllm_disagg_server_orbit.sh
'

echo "Collecting results..."
cp -r \${RUN_LOGS_DIR}/* \${WORK_DIR}/ 2>/dev/null || cp -r /tmp/run_logs/\${SLURM_JOB_ID}/* \${WORK_DIR}/ 2>/dev/null || true

echo "=== Experiment Complete: ${job_name} ==="
SBATCH_EOF

    chmod +x "$script_path"
    echo "$script_path"
}

submit_job() {
    local script_path=$1
    local job_name=$2

    if [ "$DRY_RUN" = "true" ]; then
        echo "  [DRY-RUN] Would submit: $job_name"
        return
    fi

    result=$(sbatch "$script_path" 2>&1)
    job_id=$(echo "$result" | grep -o '[0-9]*')
    echo "  SUBMITTED: $job_name -> $result"
    SUBMITTED_JOBS+=("$job_id:$job_name")
    JOB_COUNT=$((JOB_COUNT + 1))

    if [ "$SEQUENTIAL" = "true" ] && [ -n "$job_id" ]; then
        echo "  Waiting for job $job_id to complete..."
        while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
            sleep 30
        done
        echo "  Job $job_id completed."
    fi

    sleep 3
}

# =============================================================================
# Phase Functions (parameterized by model)
# =============================================================================

run_phase1() {
    local mn=$1 mp=$2 ms=$3 rd=$4
    echo ""
    echo "=== PHASE 1: Concurrency Sweep (2P2D Homo) - ${mn} ==="
    for con in 2 4 8 16 32; do
        prompts=$((con * 8))
        if [ "$prompts" -lt 64 ]; then prompts=64; fi

        local script=$(generate_job \
            "${ms}_p1_con${con}_rr" 2 2 5 "$NODES_2P2D" \
            "toy" 256 256 "$con" "$prompts" "false" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase1_Con${con}_RR"

        script=$(generate_job \
            "${ms}_p1_con${con}_mpc_rr" 2 2 5 "$NODES_2P2D" \
            "mpc_rr" 256 256 "$con" "$prompts" "false" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase1_Con${con}_MPC_RR"
    done
}

run_phase2() {
    local mn=$1 mp=$2 ms=$3 rd=$4
    echo ""
    echo "=== PHASE 2: Scheduling Algorithm Comparison - ${mn} ==="
    for policy in toy rr random po2 mpc_rr mpc_po2; do
        local script=$(generate_job \
            "${ms}_p2_${policy}_con8" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "8" "128" "false" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase2_${policy}_Con8"

        script=$(generate_job \
            "${ms}_p2_${policy}_con16" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "16" "128" "false" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase2_${policy}_Con16"
    done
}

run_phase3() {
    local mn=$1 mp=$2 ms=$3 rd=$4
    echo ""
    echo "=== PHASE 3: Heterogeneous 2P2D - ${mn} ==="
    for policy in toy po2 mpc_po2; do
        local script=$(generate_job \
            "${ms}_p3_hetero_${policy}_con8" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "8" "128" "true" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase3_Hetero_${policy}_Con8"

        script=$(generate_job \
            "${ms}_p3_hetero_${policy}_con16" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "16" "128" "true" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase3_Hetero_${policy}_Con16"
    done
}

run_phase4() {
    local mn=$1 mp=$2 ms=$3 rd=$4
    echo ""
    echo "=== PHASE 4: Heterogeneous 3P3D Scale-out - ${mn} ==="
    for policy in toy mpc_po2; do
        local script=$(generate_job \
            "${ms}_p4_3p3d_hetero_${policy}_con8" 3 3 7 "$NODES_3P3D" \
            "$policy" 256 256 "8" "128" "true" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase4_3P3D_Hetero_${policy}_Con8"

        script=$(generate_job \
            "${ms}_p4_3p3d_hetero_${policy}_con16" 3 3 7 "$NODES_3P3D" \
            "$policy" 256 256 "16" "128" "true" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase4_3P3D_Hetero_${policy}_Con16"
    done
}

run_phase5() {
    local mn=$1 mp=$2 ms=$3 rd=$4
    echo ""
    echo "=== PHASE 5: Variable ISL/OSL Study - ${mn} ==="
    for isl_osl in "64:512" "512:64" "128:256" "256:128"; do
        IFS=':' read -r isl osl <<< "$isl_osl"

        local script=$(generate_job \
            "${ms}_p5_isl${isl}_osl${osl}_rr" 2 2 5 "$NODES_2P2D" \
            "toy" "$isl" "$osl" "8" "128" "false" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase5_ISL${isl}_OSL${osl}_RR"

        script=$(generate_job \
            "${ms}_p5_isl${isl}_osl${osl}_mpc" 2 2 5 "$NODES_2P2D" \
            "mpc_po2" "$isl" "$osl" "8" "128" "false" "0.45" "64" "$mn" "$mp" "$rd")
        submit_job "$script" "${mn}_Phase5_ISL${isl}_OSL${osl}_MPC"
    done
}

# =============================================================================
# Main Loop: Iterate over models
# =============================================================================

for model_key in $MODELS_TO_RUN; do
    model_name="${MODEL_NAMES[$model_key]:-$model_key}"
    model_path="${MODEL_PATHS[$model_key]:-${MODELS_BASE_DIR}/$model_key}"
    model_short="${MODEL_SHORT[$model_key]:-$model_key}"

    RESULTS_DIR="$HOME/orbit_paper/cluster_experiments/results/${model_short}_${TIMESTAMP}"
    mkdir -p "$RESULTS_DIR/scripts" "$RESULTS_DIR/logs"

    echo ""
    echo "####################################################"
    echo "  MODEL: $model_name"
    echo "  PATH:  $model_path"
    echo "  OUT:   $RESULTS_DIR"
    echo "####################################################"

    # Verify model exists on disk
    if [[ ! -d "$model_path" ]]; then
        echo "  WARNING: Model not found at $model_path"
        echo "  Checking if HuggingFace ID is usable directly..."
    fi

    # Write manifest
    cat > "$RESULTS_DIR/MANIFEST.txt" << EOF
Orbit MPC Multi-Model Experiments
Model: $model_name ($model_path)
Docker: $DOCKER_IMAGE
Date: $(date)
Partition: $PARTITION
Nodes: $NODE_POOL

Phases: Same 5-phase matrix as Qwen2.5-14B-Instruct (40 experiments)
EOF

    case "$PHASE" in
        1) run_phase1 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR" ;;
        2) run_phase2 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR" ;;
        3) run_phase3 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR" ;;
        4) run_phase4 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR" ;;
        5) run_phase5 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR" ;;
        all)
            run_phase1 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR"
            run_phase2 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR"
            run_phase3 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR"
            run_phase4 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR"
            run_phase5 "$model_name" "$model_path" "$model_short" "$RESULTS_DIR"
            ;;
        *)
            echo "Invalid phase: $PHASE (use 1-5 or 'all')"
            exit 1
            ;;
    esac
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "  Multi-Model Experiment Submission Summary"
echo "=============================================="
echo "Total jobs: $JOB_COUNT"
echo "Models tested: $MODELS_TO_RUN"
echo ""
echo "Submitted jobs:"
for entry in "${SUBMITTED_JOBS[@]}"; do
    IFS=':' read -r jid jname <<< "$entry"
    echo "  $jid  $jname"
done
echo ""
echo "Monitor: squeue -u \$USER"
echo "=============================================="
