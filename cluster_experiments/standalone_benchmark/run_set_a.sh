#!/bin/bash
# Orbit SET A: Heterogeneity sweep
# Fixed load: 4 rps, 120 requests, max_tokens=50
# Policies: rr, lor, po2, mpc_po2
# Heterogeneity: 1x(64/64), 2x(64/32), 4x(64/16), 8x(64/8)
# Fast server: GPU 0, port 8000, max-seqs=64 (always running)
# Slow server: GPU 1, port 8002, max-seqs varies (restart each iter)

set -e

MODEL=/shared_inference/models/Qwen/Qwen3-8B
RESULTS_BASE=/shared_inference/vpolamre/orbit/results
RUN_ID=$(date +%Y%m%d_%H%M%S)_setA
RD="${RESULTS_BASE}/${RUN_ID}"
BENCH=/shared_inference/vpolamre/orbit/run_benchmark.py
ROUTER_PY=/shared_inference/vpolamre/orbit/orbit_router.py
LOG="${RD}/run.log"

mkdir -p "$RD"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "  Orbit SET A: Heterogeneity Sweep — $(date)"
echo "  Run ID : $RUN_ID"
echo "  Node   : $(hostname) ($(hostname -I | awk '{print $1}'))"
echo "  Results: $RD"
echo "========================================================"

ROUTER_PID=""

stop_router() {
    if [ -n "$ROUTER_PID" ] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        kill "$ROUTER_PID" 2>/dev/null || true; sleep 2
    fi
    ROUTER_PID=""
    fuser -k 9000/tcp 2>/dev/null || true
    sleep 1
}

start_router() {
    local policy=$1; local mpc=$2; local servers=$3
    stop_router
    CMD="python3 ${ROUTER_PY} --port 9000 --policy ${policy} ${mpc} single --servers ${servers}"
    echo "[router] $CMD"
    $CMD &
    ROUTER_PID=$!
    for i in $(seq 1 15); do
        curl -sf http://127.0.0.1:9000/health > /dev/null 2>&1 && echo "[router] up (${i}s)" && return 0
        sleep 1
    done
    echo "[router] FAILED to start"; exit 1
}

bench() {
    local label=$1 out=$2 rate=$3 n=$4
    CMD="python3 ${BENCH} --router http://127.0.0.1:9000 --model ${MODEL} \
        --arrival-rate ${rate} --num-requests ${n} --max-tokens 50 \
        --label ${label} --output ${out}"
    echo "[bench] CMD: $CMD"
    $CMD
}

dump_metrics() {
    echo "[metrics:$1]"
    curl -s http://127.0.0.1:9000/metrics | python3 -m json.tool 2>/dev/null || true
}

trap stop_router EXIT

RATE=4.0
N=120
SVRS="127.0.0.1:8000 127.0.0.1:8002"

echo ""
echo "================================================================"
echo "SET A: Heterogeneity sweep — ${RATE} rps, ${N} requests"
echo "Fast server: GPU 0 port 8000 max-seqs=64 (fixed)"
echo "Slow server: GPU 1 port 8002 max-seqs varies"
echo "================================================================"

# Start fast server once (GPU 0, port 8000, max-seqs=64) — stays up for all ratios
echo "[setup] Starting fast server (GPU 0, port 8000, max-seqs=64)..."
docker rm -f vllm_0 2>/dev/null || true
sleep 2
docker run -d --name vllm_0 --network host \
    --device /dev/kfd --device /dev/dri --group-add video \
    --ipc host --privileged -v /shared_inference:/shared_inference \
    -e HIP_VISIBLE_DEVICES=0 -e VLLM_ROCM_USE_AITER=0 \
    vllm/vllm-openai-rocm:v0.23.0 \
    /shared_inference/models/Qwen/Qwen3-8B \
    --port 8000 --tensor-parallel-size 1 \
    --max-num-seqs 64 --max-model-len 2048 \
    --gpu-memory-utilization 0.15 --dtype bfloat16

echo "[setup] Waiting for fast server (port 8000)..."
for i in $(seq 1 36); do
    curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 && echo "[setup] fast server healthy (${i}x5s)" && break
    sleep 5
done
curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1 || { echo "[setup] ERROR: fast server failed to start"; exit 1; }

for SLOW_SEQS in 64 32 16 8; do
    RATIO=$((64 / SLOW_SEQS))
    TAG="${RATIO}x"

    echo ""
    echo "---- Heterogeneity ${TAG} (fast=64, slow=${SLOW_SEQS}) — $(date) ----"

    # Restart slow server with new max-seqs
    echo "[setup] Restarting vllm_1 with max-seqs=${SLOW_SEQS}..."
    docker rm -f vllm_1 2>/dev/null || true
    sleep 3

    docker run -d --name vllm_1 --network host \
        --device /dev/kfd --device /dev/dri --group-add video \
        --ipc host --privileged -v /shared_inference:/shared_inference \
        -e HIP_VISIBLE_DEVICES=1 -e VLLM_ROCM_USE_AITER=0 \
        vllm/vllm-openai-rocm:v0.23.0 \
        /shared_inference/models/Qwen/Qwen3-8B \
        --port 8002 --tensor-parallel-size 1 \
        --max-num-seqs ${SLOW_SEQS} --max-model-len 2048 \
        --gpu-memory-utilization 0.15 --dtype bfloat16

    echo "[setup] Waiting for vllm_1 (max-seqs=${SLOW_SEQS}) to be healthy..."
    HEALTHY=0
    for i in $(seq 1 36); do
        if curl -sf http://127.0.0.1:8002/health > /dev/null 2>&1; then
            echo "[setup] vllm_1 healthy after ${i}x5s"
            HEALTHY=1
            break
        fi
        sleep 5
    done
    if [ "$HEALTHY" = "0" ]; then
        echo "[setup] ERROR: vllm_1 did not start in 3 minutes — skipping ${TAG}"
        continue
    fi

    # Warm-up: let server settle
    sleep 5

    for POLICY in rr lor po2 mpc_po2; do
        MPC_FLAG=""
        [ "$POLICY" = "mpc_po2" ] && MPC_FLAG="--enable-mpc"

        echo ""
        echo "[run] ${TAG} / ${POLICY} — $(date)"
        start_router "$POLICY" "$MPC_FLAG" "$SVRS"
        sleep 3

        LABEL="hetero_${TAG}_${POLICY}"
        OUT="${RD}/${LABEL}.json"
        bench "$LABEL" "$OUT" $RATE $N
        dump_metrics "$LABEL"
        stop_router
        sleep 2
    done

    echo "[done] ${TAG} complete — $(date)"
done

stop_router

# ================================================================
# SUMMARY
# ================================================================

echo ""
echo "================================================================"
echo "SET A SUMMARY — $(date)"
echo "================================================================"

python3 << 'PYEOF'
import json, os, sys

import os
RD = os.environ.get("RD", "")

# Find RD from log name if env not set
if not RD:
    # parse from script context — use glob
    import glob
    dirs = sorted(glob.glob("/shared_inference/vpolamre/orbit/results/*_setA"))
    if dirs:
        RD = dirs[-1]

def load(f):
    p = f"{RD}/{f}.json"
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["stats"]
    except Exception as e:
        print(f"  [warn] {f}: {e}", file=sys.stderr)
        return None

def pct(base, new, key):
    if not base or not new:
        return "n/a"
    b, n = base.get(key, 0), new.get(key, 0)
    if b == 0:
        return "n/a"
    return f"{(b - n) / b * 100:+.1f}%"

print(f"\nResults dir: {RD}\n")
print(f"{'Hetero':<8} {'Policy':<12} {'Mean(ms)':>9} {'Stdev(ms)':>10} {'P95(ms)':>9} {'P99(ms)':>9} {'N':>5}")
print("-" * 60)

for ratio in ["1x", "2x", "4x", "8x"]:
    for pol in ["rr", "lor", "po2", "mpc_po2"]:
        s = load(f"hetero_{ratio}_{pol}")
        if s:
            print(f"{ratio:<8} {pol:<12} {s['mean_ms']:>9.1f} {s['stdev_ms']:>10.1f} "
                  f"{s['p95_ms']:>9.1f} {s['p99_ms']:>9.1f} {s.get('n_ok', '?'):>5}")
    # MPC vs PO2 delta
    po2 = load(f"hetero_{ratio}_po2")
    mpc = load(f"hetero_{ratio}_mpc_po2")
    if po2 and mpc:
        print(f"         -> MPC-PO2 vs PO2: mean={pct(po2, mpc, 'mean_ms')}  "
              f"P99={pct(po2, mpc, 'p99_ms')}  stdev={pct(po2, mpc, 'stdev_ms')}")
    # LOR vs PO2 delta
    lor = load(f"hetero_{ratio}_lor")
    if lor and po2:
        print(f"         -> LOR vs PO2:     mean={pct(po2, lor, 'mean_ms')}  "
              f"P99={pct(po2, lor, 'p99_ms')}")
    print()

print(f"Full log: {RD}/run.log")
PYEOF

echo "Done: $(date)"
