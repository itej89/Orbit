#!/bin/bash
# =============================================================================
# Orbit Simulation Experiments
# Run MPC vs Baseline routing comparisons
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SIMULATION_DIR="$PROJECT_ROOT/simulation"
RESULTS_DIR="$PROJECT_ROOT/results/simulation"

mkdir -p "$RESULTS_DIR"

# =============================================================================
# Configuration
# =============================================================================

# Server configurations (delay in seconds)
PREFILL_CONFIGS=(
    "5ms_30ms:0.005:0.03"      # 6x heterogeneity
    "10ms_10ms:0.01:0.01"      # Homogeneous
    "5ms_20ms:0.005:0.02"      # 4x heterogeneity
)

DECODE_CONFIGS=(
    "10ms_50ms:0.01:0.05"      # 5x heterogeneity
    "20ms_20ms:0.02:0.02"      # Homogeneous
    "10ms_30ms:0.01:0.03"      # 3x heterogeneity
)

# Routing modes to test
ROUTING_MODES=("po2" "rr" "mpc_po2" "mpc_rr")

# Benchmark parameters
CONCURRENCY=64
REQUESTS_PER_WORKER=256
MEAN_RPS=30

# =============================================================================
# Functions
# =============================================================================

start_servers() {
    local prefill_config=$1
    local decode_config=$2
    
    IFS=':' read -r pname p1_delay p2_delay <<< "$prefill_config"
    IFS=':' read -r dname d1_delay d2_delay <<< "$decode_config"
    
    echo "Starting servers: Prefill=$pname, Decode=$dname"
    
    # Start prefill servers
    python "$SIMULATION_DIR/prefill_server.py" --port 8100 --delay "$p1_delay" &
    PREFILL1_PID=$!
    python "$SIMULATION_DIR/prefill_server.py" --port 8101 --delay "$p2_delay" &
    PREFILL2_PID=$!
    
    # Start decode servers
    python "$SIMULATION_DIR/decode_server.py" --port 8200 --token-delay "$d1_delay" &
    DECODE1_PID=$!
    python "$SIMULATION_DIR/decode_server.py" --port 8201 --token-delay "$d2_delay" &
    DECODE2_PID=$!
    
    sleep 2  # Wait for servers to start
}

stop_servers() {
    echo "Stopping servers..."
    kill $PREFILL1_PID $PREFILL2_PID $DECODE1_PID $DECODE2_PID 2>/dev/null || true
    sleep 1
}

start_router() {
    local routing_mode=$1
    
    echo "Starting router with mode: $routing_mode"
    
    # Determine if MPC should be enabled
    local mpc_flag=""
    local base_mode="po2"
    
    case $routing_mode in
        mpc_po2)
            mpc_flag="--enable-mpc"
            base_mode="po2"
            ;;
        mpc_rr)
            mpc_flag="--enable-mpc"
            base_mode="rr"
            ;;
        po2|rr)
            base_mode=$routing_mode
            ;;
    esac
    
    python "$PROJECT_ROOT/src/orbit/router.py" \
        --prefiller-hosts 127.0.0.1 127.0.0.1 \
        --prefiller-ports 8100 8101 \
        --decoder-hosts 127.0.0.1 127.0.0.1 \
        --decoder-ports 8200 8201 \
        --host 127.0.0.1 \
        --port 8000 \
        $mpc_flag &
    ROUTER_PID=$!
    
    sleep 2  # Wait for router to start
}

stop_router() {
    echo "Stopping router..."
    kill $ROUTER_PID 2>/dev/null || true
    sleep 1
}

run_benchmark() {
    local output_dir=$1
    
    echo "Running benchmark..."
    
    cd "$SIMULATION_DIR"
    
    # Modify benchmark parameters
    export CONCURRENCY=$CONCURRENCY
    export REQUESTS_PER_WORKER=$REQUESTS_PER_WORKER
    export MEAN_RPS=$MEAN_RPS
    
    python benchmark.py
    
    # Move results
    mv requests.csv "$output_dir/requests.csv"
    mv metrics.csv "$output_dir/metrics.csv"
    
    cd -
}

analyze_results() {
    local results_dir=$1
    
    echo "Analyzing results in $results_dir"
    
    python3 << EOF
import pandas as pd
import json

requests = pd.read_csv("$results_dir/requests.csv")
metrics = pd.read_csv("$results_dir/metrics.csv")

# Latency statistics
latency_stats = {
    "mean_ms": requests["latency"].mean() * 1000,
    "p50_ms": requests["latency"].quantile(0.5) * 1000,
    "p99_ms": requests["latency"].quantile(0.99) * 1000,
    "std_ms": requests["latency"].std() * 1000,
    "total_requests": len(requests),
    "success_rate": (requests["status"] == 200).mean() * 100,
}

# Save summary
with open("$results_dir/summary.json", "w") as f:
    json.dump(latency_stats, f, indent=2)

print(json.dumps(latency_stats, indent=2))
EOF
}

# =============================================================================
# Main Experiment Loop
# =============================================================================

echo "=============================================="
echo "Orbit Simulation Experiments"
echo "=============================================="

for prefill_config in "${PREFILL_CONFIGS[@]}"; do
    for decode_config in "${DECODE_CONFIGS[@]}"; do
        IFS=':' read -r pname _ _ <<< "$prefill_config"
        IFS=':' read -r dname _ _ <<< "$decode_config"
        
        for routing_mode in "${ROUTING_MODES[@]}"; do
            exp_name="p_${pname}_d_${dname}_${routing_mode}"
            exp_dir="$RESULTS_DIR/$exp_name"
            
            echo ""
            echo "=============================================="
            echo "Experiment: $exp_name"
            echo "=============================================="
            
            mkdir -p "$exp_dir"
            
            # Run experiment
            start_servers "$prefill_config" "$decode_config"
            start_router "$routing_mode"
            run_benchmark "$exp_dir"
            stop_router
            stop_servers
            
            # Analyze
            analyze_results "$exp_dir"
            
            echo "Completed: $exp_name"
        done
    done
done

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results in: $RESULTS_DIR"
echo "=============================================="
