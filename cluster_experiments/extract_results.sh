#!/bin/bash
# Extract benchmark metrics from all experiment results
# Usage: ./extract_results.sh <results_dir> [output_csv]

RESULTS_DIR="${1:-.}"
OUTPUT="${2:-results_summary.csv}"

echo "Model,Phase,Experiment,Policy,Concurrency,ISL,OSL,Hetero,Topology,MeanTTFT_ms,MedianTTFT_ms,P99TTFT_ms,MeanTPOT_ms,MedianTPOT_ms,P99TPOT_ms,MeanITL_ms,MedianITL_ms,P99ITL_ms,Throughput_req_s,OutputTok_s,Duration_s,SuccessReqs,FailedReqs" > "$OUTPUT"

for exp_dir in "$RESULTS_DIR"/*/; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    [[ "$exp_name" == "logs" || "$exp_name" == "scripts" ]] && continue

    bench_file=$(ls "$exp_dir"/benchmark_*.log 2>/dev/null | head -1)
    [ -z "$bench_file" ] && continue

    # Parse experiment name to extract metadata
    model=""
    phase=""
    policy=""
    concurrency=""
    isl=""
    osl=""
    hetero="false"
    topology="2P2D"

    if [[ "$exp_name" =~ _p1_ ]]; then
        phase="P1_ConcurrencySweep"
        if [[ "$exp_name" =~ _mpc_rr ]]; then policy="mpc_rr"; else policy="rr"; fi
        concurrency=$(echo "$exp_name" | grep -oP 'con\K[0-9]+')
        isl=256; osl=256
    elif [[ "$exp_name" =~ _p2_ ]]; then
        phase="P2_AlgoComparison"
        policy=$(echo "$exp_name" | sed 's/.*_p2_//' | sed 's/_con[0-9]*//')
        concurrency=$(echo "$exp_name" | grep -oP 'con\K[0-9]+')
        isl=256; osl=256
    elif [[ "$exp_name" =~ _p3_ ]]; then
        phase="P3_Hetero2P2D"
        hetero="true"
        policy=$(echo "$exp_name" | sed 's/.*_p3_hetero_//' | sed 's/_con[0-9]*//')
        concurrency=$(echo "$exp_name" | grep -oP 'con\K[0-9]+')
        isl=256; osl=256
    elif [[ "$exp_name" =~ _p4_ ]]; then
        phase="P4_Hetero3P3D"
        hetero="true"
        topology="3P3D"
        policy=$(echo "$exp_name" | sed 's/.*_p4_3p3d_hetero_//' | sed 's/_con[0-9]*//')
        concurrency=$(echo "$exp_name" | grep -oP 'con\K[0-9]+')
        isl=256; osl=256
    elif [[ "$exp_name" =~ _p5_ ]]; then
        phase="P5_VariableISLOSL"
        isl=$(echo "$exp_name" | grep -oP 'isl\K[0-9]+')
        osl=$(echo "$exp_name" | grep -oP 'osl\K[0-9]+')
        if [[ "$exp_name" =~ _mpc ]]; then policy="mpc_po2"; else policy="rr"; fi
        concurrency=8
    fi

    # Extract metrics from benchmark log
    mean_ttft=$(grep "Mean TTFT" "$bench_file" | awk '{print $NF}')
    median_ttft=$(grep "Median TTFT" "$bench_file" | awk '{print $NF}')
    p99_ttft=$(grep "P99 TTFT" "$bench_file" | awk '{print $NF}')
    mean_tpot=$(grep "Mean TPOT" "$bench_file" | awk '{print $NF}')
    median_tpot=$(grep "Median TPOT" "$bench_file" | awk '{print $NF}')
    p99_tpot=$(grep "P99 TPOT" "$bench_file" | awk '{print $NF}')
    mean_itl=$(grep "Mean ITL" "$bench_file" | awk '{print $NF}')
    median_itl=$(grep "Median ITL" "$bench_file" | awk '{print $NF}')
    p99_itl=$(grep "P99 ITL" "$bench_file" | awk '{print $NF}')
    throughput=$(grep "Request throughput" "$bench_file" | awk '{print $NF}')
    output_toks=$(grep "Output token throughput" "$bench_file" | head -1 | awk '{print $NF}')
    duration=$(grep "Benchmark duration" "$bench_file" | awk '{print $NF}')
    success=$(grep "Successful requests" "$bench_file" | awk '{print $NF}')
    failed=$(grep "Failed requests" "$bench_file" | awk '{print $NF}')

    model_name=$(echo "$exp_name" | sed 's/_p[1-5]_.*//')

    echo "${model_name},${phase},${exp_name},${policy},${concurrency},${isl},${osl},${hetero},${topology},${mean_ttft},${median_ttft},${p99_ttft},${mean_tpot},${median_tpot},${p99_tpot},${mean_itl},${median_itl},${p99_itl},${throughput},${output_toks},${duration},${success},${failed}" >> "$OUTPUT"
done

echo "Extracted $(wc -l < "$OUTPUT") entries (including header) to $OUTPUT"
sort -t',' -k2,2 -k4,4 -k5,5n "$OUTPUT" > "${OUTPUT}.sorted"
mv "${OUTPUT}.sorted" "$OUTPUT"
