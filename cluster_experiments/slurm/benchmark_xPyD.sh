#!/bin/bash
# =============================================================================
# Benchmark for Orbit MPC Experiments
# Configurable ISL/OSL/concurrency/prompts via environment variables
# =============================================================================

timestamp=$(date "+%Y%m%d_%H%M%S")
LOG="/run_logs/${SLURM_JOB_ID}/benchmark_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}"

BENCH_ISL="${BENCH_ISL:-128}"
BENCH_OSL="${BENCH_OSL:-128}"
BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-4}"
BENCH_PROMPTS="${BENCH_PROMPTS:-32}"

echo "==== Benchmark Serving Test ${LOG} ====="
echo "UTC Time: $(TZ=UTC date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "PST Time: $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "ISL=${BENCH_ISL} OSL=${BENCH_OSL} CON=${BENCH_CONCURRENCY} PROMPTS=${BENCH_PROMPTS}" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo "MPC_ENABLED=${MPC_ENABLED}" | tee -a ${LOG}_CONCURRENCY.log >/dev/null
echo ""

# Run benchmark across multiple concurrency levels to see scaling behavior
IFS=',' read -ra CON_LEVELS <<< "$BENCH_CONCURRENCY"

for con in "${CON_LEVELS[@]}"; do
    p=$BENCH_PROMPTS
    if [ "$p" -lt "$((con * 4))" ]; then
        p=$((con * 4))
    fi

    echo "[RUNNING] isl=$BENCH_ISL osl=$BENCH_OSL concurrency=$con prompts=$p"
    vllm bench serve \
        --model $MODEL_PATH \
        --backend vllm \
        --host 127.0.0.1 \
        --port ${SERVER_PORT:-2584} \
        --dataset-name "random" \
        --random-input-len $BENCH_ISL \
        --random-output-len $BENCH_OSL \
        --random-prefix-len 0 \
        --num-prompts $p \
        --request-rate "inf" \
        --ignore-eos \
        --max-concurrency $con \
        2>&1 | tee -a ${LOG}_CONCURRENCY.log >/dev/null

    sleep 5
done

echo "=== Benchmark Complete ==="
