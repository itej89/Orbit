#!/bin/bash
# =============================================================================
# Collect and format all experiment results
# =============================================================================

RESULTS_DIR="${1:-$(ls -td ~/orbit_paper/cluster_experiments/results/comprehensive_* 2>/dev/null | head -1)}"

if [ -z "$RESULTS_DIR" ] || [ ! -d "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

echo "=============================================="
echo "  Orbit MPC Experiment Results"
echo "  Source: $RESULTS_DIR"
echo "  Date: $(date)"
echo "=============================================="
echo ""

# Collect all results into structured format
echo "=== PHASE 1: Concurrency Sweep (2P2D Homogeneous) ==="
echo "  RR baseline vs MPC_RR - ISL=256 OSL=256"
echo ""
printf "%-6s | %-6s | %11s | %9s | %8s | %8s | %8s\n" "CON" "POLICY" "TPUT(tok/s)" "TPOT_mean" "TPOT_P99" "ITL_P99" "TTFT_P99"
echo "-------|--------|-------------|-----------|----------|----------|----------"

for con in 2 4 8 16 32; do
    for suffix in "rr" "mpc_rr"; do
        d="$RESULTS_DIR/p1_con${con}_${suffix}"
        bfile=$(ls $d/benchmark_*_CONCURRENCY.log 2>/dev/null | tail -1)
        if [ -n "$bfile" ]; then
            tput=$(grep "Output token throughput" $bfile | head -1 | awk '{print $NF}')
            tpot_m=$(grep "Mean TPOT" $bfile | tail -1 | awk '{print $NF}')
            tpot_p99=$(grep "P99 TPOT" $bfile | tail -1 | awk '{print $NF}')
            itl_p99=$(grep "P99 ITL" $bfile | tail -1 | awk '{print $NF}')
            ttft_p99=$(grep "P99 TTFT" $bfile | tail -1 | awk '{print $NF}')
            printf "%-6s | %-6s | %11s | %9s | %8s | %8s | %8s\n" "$con" "$suffix" "$tput" "$tpot_m" "$tpot_p99" "$itl_p99" "$ttft_p99"
        fi
    done
done

echo ""
echo "=== PHASE 2: Scheduling Algorithm Comparison (2P2D Homogeneous) ==="
echo "  ISL=256 OSL=256 Prompts=128"
echo ""
printf "%-10s | %-6s | %11s | %9s | %8s | %8s\n" "POLICY" "CON" "TPUT(tok/s)" "TPOT_mean" "TPOT_P99" "ITL_P99"
echo "-----------|--------|-------------|-----------|----------|----------"

for policy in toy rr random po2 mpc_rr mpc_po2; do
    for con in 8 16; do
        d="$RESULTS_DIR/p2_${policy}_con${con}"
        bfile=$(ls $d/benchmark_*_CONCURRENCY.log 2>/dev/null | tail -1)
        if [ -n "$bfile" ]; then
            tput=$(grep "Output token throughput" $bfile | head -1 | awk '{print $NF}')
            tpot_m=$(grep "Mean TPOT" $bfile | tail -1 | awk '{print $NF}')
            tpot_p99=$(grep "P99 TPOT" $bfile | tail -1 | awk '{print $NF}')
            itl_p99=$(grep "P99 ITL" $bfile | tail -1 | awk '{print $NF}')
            printf "%-10s | %-6s | %11s | %9s | %8s | %8s\n" "$policy" "$con" "$tput" "$tpot_m" "$tpot_p99" "$itl_p99"
        fi
    done
done

echo ""
echo "=== PHASE 3: Heterogeneous 2P2D ==="
echo "  P2/D2 throttled (gpu_mem=0.45, max_seqs=64), ISL=256 OSL=256"
echo ""
printf "%-10s | %-6s | %11s | %9s | %8s | %8s\n" "POLICY" "CON" "TPUT(tok/s)" "TPOT_mean" "TPOT_P99" "ITL_P99"
echo "-----------|--------|-------------|-----------|----------|----------"

for policy in toy po2 mpc_po2; do
    for con in 8 16; do
        d="$RESULTS_DIR/p3_hetero_${policy}_con${con}"
        bfile=$(ls $d/benchmark_*_CONCURRENCY.log 2>/dev/null | tail -1)
        if [ -n "$bfile" ]; then
            tput=$(grep "Output token throughput" $bfile | head -1 | awk '{print $NF}')
            tpot_m=$(grep "Mean TPOT" $bfile | tail -1 | awk '{print $NF}')
            tpot_p99=$(grep "P99 TPOT" $bfile | tail -1 | awk '{print $NF}')
            itl_p99=$(grep "P99 ITL" $bfile | tail -1 | awk '{print $NF}')
            printf "%-10s | %-6s | %11s | %9s | %8s | %8s\n" "$policy" "$con" "$tput" "$tpot_m" "$tpot_p99" "$itl_p99"
        fi
    done
done

echo ""
echo "=== PHASE 4: Heterogeneous 3P3D Scale-out ==="
echo "  3P+3D with 2 throttled each, ISL=256 OSL=256"
echo ""
printf "%-10s | %-6s | %11s | %9s | %8s | %8s\n" "POLICY" "CON" "TPUT(tok/s)" "TPOT_mean" "TPOT_P99" "ITL_P99"
echo "-----------|--------|-------------|-----------|----------|----------"

for policy in toy mpc_po2; do
    for con in 8 16; do
        d="$RESULTS_DIR/p4_3p3d_hetero_${policy}_con${con}"
        bfile=$(ls $d/benchmark_*_CONCURRENCY.log 2>/dev/null | tail -1)
        if [ -n "$bfile" ]; then
            tput=$(grep "Output token throughput" $bfile | head -1 | awk '{print $NF}')
            tpot_m=$(grep "Mean TPOT" $bfile | tail -1 | awk '{print $NF}')
            tpot_p99=$(grep "P99 TPOT" $bfile | tail -1 | awk '{print $NF}')
            itl_p99=$(grep "P99 ITL" $bfile | tail -1 | awk '{print $NF}')
            printf "%-10s | %-6s | %11s | %9s | %8s | %8s\n" "$policy" "$con" "$tput" "$tpot_m" "$tpot_p99" "$itl_p99"
        fi
    done
done

echo ""
echo "=== PHASE 5: Variable ISL/OSL (2P2D Homogeneous) ==="
echo "  RR vs MPC_PO2 at Con=8"
echo ""
printf "%-16s | %-10s | %11s | %9s | %8s | %8s\n" "WORKLOAD" "POLICY" "TPUT(tok/s)" "TPOT_mean" "TPOT_P99" "ITL_P99"
echo "-----------------|-----------|-------------|-----------|----------|----------"

for isl_osl in "64_512" "512_64" "128_256" "256_128"; do
    IFS='_' read -r isl osl <<< "$isl_osl"
    for suffix in "rr" "mpc"; do
        d="$RESULTS_DIR/p5_isl${isl}_osl${osl}_${suffix}"
        bfile=$(ls $d/benchmark_*_CONCURRENCY.log 2>/dev/null | tail -1)
        if [ -n "$bfile" ]; then
            tput=$(grep "Output token throughput" $bfile | head -1 | awk '{print $NF}')
            tpot_m=$(grep "Mean TPOT" $bfile | tail -1 | awk '{print $NF}')
            tpot_p99=$(grep "P99 TPOT" $bfile | tail -1 | awk '{print $NF}')
            itl_p99=$(grep "P99 ITL" $bfile | tail -1 | awk '{print $NF}')
            printf "ISL=%3s OSL=%3s | %-10s | %11s | %9s | %8s | %8s\n" "$isl" "$osl" "$suffix" "$tput" "$tpot_m" "$tpot_p99" "$itl_p99"
        fi
    done
done

echo ""
echo "=============================================="
echo "  Analysis Summary"
echo "=============================================="
