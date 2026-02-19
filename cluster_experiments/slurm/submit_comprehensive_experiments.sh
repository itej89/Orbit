#!/bin/bash
# =============================================================================
# Orbit MPC - Comprehensive Experiment Suite for Qwen2.5-14B-Instruct
# =============================================================================
#
# Experiment Matrix:
#   Phase 1: Concurrency Sweep (2P2D homo, RR baseline vs MPC+RR)
#   Phase 2: Scheduling Algorithm Comparison (2P2D homo, all 5 policies)
#   Phase 3: Heterogeneous 2P2D (throttled nodes, RR vs PO2 vs MPC_PO2)
#   Phase 4: Heterogeneous 3P3D Scale-out (RR vs MPC_PO2)
#   Phase 5: Variable ISL/OSL study (2P2D, RR vs MPC_PO2)
#
# Usage:
#   ./submit_comprehensive_experiments.sh --phase 1       # Run phase 1 only
#   ./submit_comprehensive_experiments.sh --phase all     # Run all phases
#   ./submit_comprehensive_experiments.sh --dry-run       # Generate scripts only
#   ./submit_comprehensive_experiments.sh --sequential    # Wait for each job
# =============================================================================

set -e

DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch-private:vllm-v0.14.0_amd_dev_aiter_nixl_ravgupta}"
PARTITION="${PARTITION:-amd-rccl}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${RESULTS_DIR:-$HOME/orbit_paper/cluster_experiments/results/comprehensive_${TIMESTAMP}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NODE_POOL="useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027,useocpm2m-097-042,useocpm2m-097-043,useocpm2m-097-047"

MODEL_NAME="Qwen14B"
MODEL_PATH="Qwen/Qwen2.5-14B-Instruct"

NODES_2P2D="useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027"
NODES_3P3D="useocpm2m-097-019,useocpm2m-097-020,useocpm2m-097-023,useocpm2m-097-025,useocpm2m-097-027,useocpm2m-097-042,useocpm2m-097-043"

# =============================================================================
# Argument Parsing
# =============================================================================

PHASE="all"
DRY_RUN=false
SEQUENTIAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --sequential) SEQUENTIAL=true; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "=============================================="
echo "  Orbit MPC Comprehensive Experiments"
echo "=============================================="
echo "Model:     $MODEL_PATH"
echo "Phase:     $PHASE"
echo "Results:   $RESULTS_DIR"
echo "Dry run:   $DRY_RUN"
echo "Sequential: $SEQUENTIAL"
echo "=============================================="

mkdir -p "$RESULTS_DIR/scripts" "$RESULTS_DIR/logs"

# Write experiment manifest
cat > "$RESULTS_DIR/MANIFEST.txt" << EOF
Orbit MPC Comprehensive Experiments
Model: $MODEL_PATH ($MODEL_NAME)
Docker: $DOCKER_IMAGE
Date: $(date)
Partition: $PARTITION
Nodes: $NODE_POOL

Phase 1: Concurrency Sweep (2P2D homo) - RR baseline vs MPC_RR
  Concurrency levels: 2,4,8,16,32
  ISL=256 OSL=256 Prompts=256

Phase 2: Scheduling Algorithm Comparison (2P2D homo)
  Policies: toy(rr), rr, random, po2, mpc_rr, mpc_po2
  ISL=256 OSL=256 Con=8,16 Prompts=128

Phase 3: Heterogeneous 2P2D (capacity-imbalanced nodes)
  P2 and D2 throttled: gpu_mem=0.45, max_seqs=64
  Policies: toy, po2, mpc_po2
  ISL=256 OSL=256 Con=8,16 Prompts=128

Phase 4: Heterogeneous 3P3D Scale-out
  P2,P3 and D2,D3 throttled
  Policies: toy, mpc_po2
  ISL=256 OSL=256 Con=8,16 Prompts=128

Phase 5: Variable ISL/OSL (2P2D homo)
  Workloads: ISL=64/OSL=512, ISL=512/OSL=64, ISL=256/OSL=256
  Policies: toy, mpc_po2
  Con=8 Prompts=128
EOF

JOB_COUNT=0
SUBMITTED_JOBS=()

# =============================================================================
# Job Generation
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
mkdir -p /tmp/run_logs/\${SLURM_JOB_ID}

echo "=========================================="
echo "  Orbit Experiment: ${job_name}"
echo "=========================================="
echo "Job: \${SLURM_JOB_ID}"
echo "Nodes: \${SLURM_JOB_NODELIST}"
echo "Config: ${num_prefill}P / ${num_decode}D"
echo "Policy: ${proxy_policy}"
echo "Hetero: ${hetero_mode}"
echo "Benchmark: ISL=${bench_isl} OSL=${bench_osl} CON=${bench_concurrency} PROMPTS=${bench_prompts}"

# Resolve node IPs
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

echo "Pulling image..."
srun --ntasks-per-node=1 bash -c "docker pull \${DOCKER_IMAGE}" || true
srun --ntasks-per-node=1 bash -c "mkdir -p /tmp/run_logs/\${SLURM_JOB_ID}"

echo "Launching servers..."
srun --ntasks-per-node=1 --export=ALL bash -c '
    NODE_RANK=\$SLURM_PROCID
    HOST_NIXL_PATH="\${NIXL_COOKBOOK_PATH}"
    HOST_ORBIT_PATH="\${ORBIT_CODE_DIR}"

    PORT_CONFIG_HOST="/shared_inference/ravgupta/orbit_ports"
    mkdir -p \${PORT_CONFIG_HOST} 2>/dev/null || true

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
        -v /tmp/run_logs:/run_logs \
        -v \${PORT_CONFIG_HOST}:/port_config \
        -v /shared_inference:/shared_inference \
        -v \${HOST_ORBIT_PATH}:/opt/orbit:ro \
        -v \${HOST_NIXL_PATH}:/app/nixl_cookbook:ro \
        \${DOCKER_IMAGE} \
        bash /app/nixl_cookbook/vllm_disagg_server_orbit.sh
'

echo "Collecting results..."
cp -r /tmp/run_logs/\${SLURM_JOB_ID}/* \${WORK_DIR}/ 2>/dev/null || true

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
# Phase 1: Concurrency Sweep (2P2D homo)
# =============================================================================

run_phase1() {
    echo ""
    echo "=== PHASE 1: Concurrency Sweep (2P2D Homo) ==="
    echo "  Tests how MPC scales with increasing request concurrency"
    echo "  All servers identical (no heterogeneity)"
    echo ""

    for con in 2 4 8 16 32; do
        prompts=$((con * 8))
        if [ "$prompts" -lt 64 ]; then prompts=64; fi

        # Baseline: toy proxy (RR)
        local script=$(generate_job \
            "p1_con${con}_rr" 2 2 5 "$NODES_2P2D" \
            "toy" 256 256 "$con" "$prompts" "false")
        submit_job "$script" "Phase1_Con${con}_RR"

        # MPC+RR
        script=$(generate_job \
            "p1_con${con}_mpc_rr" 2 2 5 "$NODES_2P2D" \
            "mpc_rr" 256 256 "$con" "$prompts" "false")
        submit_job "$script" "Phase1_Con${con}_MPC_RR"
    done
}

# =============================================================================
# Phase 2: Scheduling Algorithm Comparison (2P2D homo)
# =============================================================================

run_phase2() {
    echo ""
    echo "=== PHASE 2: Scheduling Algorithm Comparison ==="
    echo "  Compares all routing policies at medium/high load"
    echo ""

    for policy in toy rr random po2 mpc_rr mpc_po2; do
        local script=$(generate_job \
            "p2_${policy}_con8" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "8" "128" "false")
        submit_job "$script" "Phase2_${policy}_Con8"

        script=$(generate_job \
            "p2_${policy}_con16" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "16" "128" "false")
        submit_job "$script" "Phase2_${policy}_Con16"
    done
}

# =============================================================================
# Phase 3: Heterogeneous 2P2D
# =============================================================================

run_phase3() {
    echo ""
    echo "=== PHASE 3: Heterogeneous 2P2D ==="
    echo "  P2 and D2 throttled (gpu_mem=0.45, max_seqs=64)"
    echo "  This is where MPC should clearly outperform naive RR"
    echo ""

    for policy in toy po2 mpc_po2; do
        local script=$(generate_job \
            "p3_hetero_${policy}_con8" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "8" "128" "true" "0.45" "64")
        submit_job "$script" "Phase3_Hetero_${policy}_Con8"

        script=$(generate_job \
            "p3_hetero_${policy}_con16" 2 2 5 "$NODES_2P2D" \
            "$policy" 256 256 "16" "128" "true" "0.45" "64")
        submit_job "$script" "Phase3_Hetero_${policy}_Con16"
    done
}

# =============================================================================
# Phase 4: Heterogeneous 3P3D Scale-out
# =============================================================================

run_phase4() {
    echo ""
    echo "=== PHASE 4: Heterogeneous 3P3D Scale-out ==="
    echo "  3 prefill + 3 decode, with 2 of each throttled"
    echo ""

    for policy in toy mpc_po2; do
        local script=$(generate_job \
            "p4_3p3d_hetero_${policy}_con8" 3 3 7 "$NODES_3P3D" \
            "$policy" 256 256 "8" "128" "true" "0.45" "64")
        submit_job "$script" "Phase4_3P3D_Hetero_${policy}_Con8"

        script=$(generate_job \
            "p4_3p3d_hetero_${policy}_con16" 3 3 7 "$NODES_3P3D" \
            "$policy" 256 256 "16" "128" "true" "0.45" "64")
        submit_job "$script" "Phase4_3P3D_Hetero_${policy}_Con16"
    done
}

# =============================================================================
# Phase 5: Variable ISL/OSL
# =============================================================================

run_phase5() {
    echo ""
    echo "=== PHASE 5: Variable ISL/OSL Study ==="
    echo "  Tests how routing adapts to different request shapes"
    echo ""

    for isl_osl in "64:512" "512:64" "128:256" "256:128"; do
        IFS=':' read -r isl osl <<< "$isl_osl"

        local script=$(generate_job \
            "p5_isl${isl}_osl${osl}_rr" 2 2 5 "$NODES_2P2D" \
            "toy" "$isl" "$osl" "8" "128" "false")
        submit_job "$script" "Phase5_ISL${isl}_OSL${osl}_RR"

        script=$(generate_job \
            "p5_isl${isl}_osl${osl}_mpc" 2 2 5 "$NODES_2P2D" \
            "mpc_po2" "$isl" "$osl" "8" "128" "false")
        submit_job "$script" "Phase5_ISL${isl}_OSL${osl}_MPC"
    done
}

# =============================================================================
# Execute Phases
# =============================================================================

case "$PHASE" in
    1) run_phase1 ;;
    2) run_phase2 ;;
    3) run_phase3 ;;
    4) run_phase4 ;;
    5) run_phase5 ;;
    all)
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        run_phase5
        ;;
    *)
        echo "Invalid phase: $PHASE (use 1-5 or 'all')"
        exit 1
        ;;
esac

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "  Experiment Submission Summary"
echo "=============================================="
echo "Total jobs: $JOB_COUNT"
echo "Results:    $RESULTS_DIR"
echo ""
echo "Submitted jobs:"
for entry in "${SUBMITTED_JOBS[@]}"; do
    IFS=':' read -r jid jname <<< "$entry"
    echo "  $jid  $jname"
done
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    $RESULTS_DIR/logs/"
echo "=============================================="

# Save job list for later analysis
cat > "$RESULTS_DIR/JOBS.txt" << EOF
# Job ID : Job Name
$(for entry in "${SUBMITTED_JOBS[@]}"; do echo "$entry"; done)
EOF
