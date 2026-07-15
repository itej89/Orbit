#!/bin/bash
# ======================================================================
# Orbit Full Replication Study
# ======================================================================
# SET A:  1x/2x/4x/8x heterogeneity × {RR, LOR, PO2, MPC-PO2, Kalman-PO2}
#         120 req @ 4 rps per cell. Fresh server restart per policy run
#         (most rigorous: eliminates any KV-cache or thermal carry-over).
#
# SET B:  10K request convergence sweep at 8x heterogeneity
#         Policies: PO2, MPC-PO2, Kalman-PO2. Fresh server restart per policy.
#
# Methodological guarantees:
#   - Router PID killed and port flushed before every run
#   - Both vLLM servers docker rm -f + fresh docker run before every policy
#   - Health-checked before benchmark starts
#   - Poisson arrivals (expovariate), max_tokens=50, Qwen3-8B bfloat16
# ======================================================================

set -euo pipefail

MODEL=/shared_inference/models/Qwen/Qwen3-8B
RESULTS_BASE=/shared_inference/vpolamre/orbit/results
RUN_ID=$(date +%Y%m%d_%H%M%S)_replication
RD="${RESULTS_BASE}/${RUN_ID}"
BENCH=/shared_inference/vpolamre/orbit/run_benchmark.py
ROUTER_PY=/shared_inference/vpolamre/orbit/orbit_router_kalman.py
LOG="${RD}/run.log"

mkdir -p "$RD"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================================"
echo "  Orbit Full Replication — $(date)"
echo "  Run ID  : $RUN_ID"
echo "  Node    : $(hostname)"
echo "  Router  : $ROUTER_PY"
echo "  Results : $RD"
echo "  SET A   : 1x/2x/4x/8x × 5 policies × 120 req @ 4 rps"
echo "  SET B   : 8x × 3 policies × 10000 req @ 4 rps"
echo "  Method  : Fresh docker containers before every policy run"
echo "========================================================================"

ROUTER_PID=""

# ── helpers ──────────────────────────────────────────────────────────────

stop_router() {
    if [ -n "$ROUTER_PID" ] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        kill "$ROUTER_PID" 2>/dev/null || true
        sleep 2
    fi
    ROUTER_PID=""
    fuser -k 9000/tcp 2>/dev/null || true
    sleep 1
}

start_router() {
    local policy=$1 mpc_flag=$2
    stop_router
    local CMD="python3 ${ROUTER_PY} --port 9000 --policy ${policy} ${mpc_flag} single \
        --servers 127.0.0.1:8000 127.0.0.1:8002"
    echo "[router] $CMD"
    $CMD &
    ROUTER_PID=$!
    for i in $(seq 1 20); do
        curl -sf http://127.0.0.1:9000/health >/dev/null 2>&1 \
            && echo "[router] up (${i}s)" && return 0
        sleep 1
    done
    echo "[router] FAILED to start after 20s"; exit 1
}

start_servers() {
    local fast_seqs=$1 slow_seqs=$2
    echo "[servers] Starting fast(${fast_seqs} seqs) + slow(${slow_seqs} seqs)..."
    docker rm -f vllm_fast vllm_slow 2>/dev/null || true
    sleep 3

    docker run -d --name vllm_fast --network host \
        --device /dev/kfd --device /dev/dri --group-add video \
        --ipc host --privileged -v /shared_inference:/shared_inference \
        -e HIP_VISIBLE_DEVICES=0 -e VLLM_ROCM_USE_AITER=0 \
        vllm/vllm-openai-rocm:v0.23.0 \
        "${MODEL}" --port 8000 --tensor-parallel-size 1 \
        --max-num-seqs ${fast_seqs} --max-model-len 2048 \
        --gpu-memory-utilization 0.15 --dtype bfloat16

    docker run -d --name vllm_slow --network host \
        --device /dev/kfd --device /dev/dri --group-add video \
        --ipc host --privileged -v /shared_inference:/shared_inference \
        -e HIP_VISIBLE_DEVICES=1 -e VLLM_ROCM_USE_AITER=0 \
        vllm/vllm-openai-rocm:v0.23.0 \
        "${MODEL}" --port 8002 --tensor-parallel-size 1 \
        --max-num-seqs ${slow_seqs} --max-model-len 2048 \
        --gpu-memory-utilization 0.15 --dtype bfloat16

    local HEALTHY=0
    for i in $(seq 1 72); do   # up to 6 min
        local ok0 ok2
        ok0=$(curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && echo 1 || echo 0)
        ok2=$(curl -sf http://127.0.0.1:8002/health >/dev/null 2>&1 && echo 1 || echo 0)
        if [ "$ok0" = "1" ] && [ "$ok2" = "1" ]; then
            echo "[servers] both healthy (${i}×5s)"
            HEALTHY=1; break
        fi
        sleep 5
    done
    [ "$HEALTHY" = "0" ] && echo "[servers] ERROR: did not start in 6 min" && exit 1
    sleep 5   # extra settle time
}

stop_servers() {
    docker rm -f vllm_fast vllm_slow 2>/dev/null || true
    sleep 3
}

# Dump router /metrics after each run for audit trail
dump_metrics() {
    local label=$1
    echo "[metrics:${label}]"
    curl -s http://127.0.0.1:9000/metrics | python3 -m json.tool 2>/dev/null || true
}

# Run one benchmark cell: starts servers fresh, runs policy, stops servers
run_cell() {
    local label=$1 policy=$2 mpc_flag=$3 fast_seqs=$4 slow_seqs=$5 rate=$6 nreq=$7

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  CELL: ${label}  policy=${policy}  fast=${fast_seqs}  slow=${slow_seqs}"
    echo "  ${rate} rps  ${nreq} req  $(date)"
    echo "────────────────────────────────────────────────────────────"

    start_servers "$fast_seqs" "$slow_seqs"
    start_router  "$policy" "$mpc_flag"
    sleep 3

    python3 "${BENCH}" \
        --router http://127.0.0.1:9000 \
        --model  "${MODEL}" \
        --arrival-rate "${rate}" \
        --num-requests "${nreq}" \
        --max-tokens 50 \
        --label  "${label}" \
        --output "${RD}/${label}.json"

    dump_metrics "$label"
    stop_router
    stop_servers
    echo "[cell done] ${label} — $(date)"
}

trap 'stop_router; stop_servers' EXIT

# ======================================================================
# SET A: Heterogeneity sweep (120 req)
# ======================================================================
echo ""
echo "======================================================================"
echo "SET A — Heterogeneity Sweep  (120 req @ 4 rps per cell)"
echo "======================================================================"

for SLOW_SEQS in 64 32 16 8; do
    RATIO=$((64 / SLOW_SEQS))x
    echo ""
    echo "══ Ratio ${RATIO} (fast=64, slow=${SLOW_SEQS}) ══"

    for POLICY in rr lor po2 mpc_po2 kalman_po2; do
        MPC_FLAG=""
        [ "$POLICY" = "mpc_po2" ] && MPC_FLAG="--enable-mpc"

        LABEL="seta_${RATIO}_${POLICY}"
        run_cell "$LABEL" "$POLICY" "$MPC_FLAG" 64 "$SLOW_SEQS" 4.0 120
    done
done

# ======================================================================
# SET B: 10K convergence sweep (PO2, MPC-PO2, Kalman-PO2 at 8x)
# ======================================================================
echo ""
echo "======================================================================"
echo "SET B — 10K Convergence Sweep  (10000 req @ 4 rps, 8x hetero)"
echo "======================================================================"

for POLICY in po2 mpc_po2 kalman_po2; do
    MPC_FLAG=""
    [ "$POLICY" = "mpc_po2" ] && MPC_FLAG="--enable-mpc"

    LABEL="setb_8x_${POLICY}_10k"
    run_cell "$LABEL" "$POLICY" "$MPC_FLAG" 64 8 4.0 10000
done

# ======================================================================
# ANALYSIS
# ======================================================================
echo ""
echo "======================================================================"
echo "ANALYSIS — $(date)"
echo "======================================================================"

python3 << PYEOF
import json, statistics, os, sys

RD = "${RD}"

# ── helpers ──────────────────────────────────────────────────────────────

def load(label):
    p = f"{RD}/{label}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    raw = [r for r in d.get("raw", []) if r.get("ok")]
    if not raw:
        return None
    lats = sorted([r["latency"] for r in raw])
    n = len(lats)
    return {
        "n":     n,
        "mean":  statistics.mean(lats),
        "stdev": statistics.stdev(lats) if n > 1 else 0.0,
        "p50":   lats[n // 2],
        "p95":   lats[int(0.95 * n)],
        "p99":   lats[int(0.99 * n)],
        "lats":  lats,
        "raw":   d.get("raw", []),
    }

def row(s):
    if s is None:
        return "         ---"
    return (f"  mean={s['mean']:6.1f}  stdev={s['stdev']:6.1f}"
            f"  p95={s['p95']:6.1f}  p99={s['p99']:7.1f}  n={s['n']}")

def vs(base, new):
    if base is None or new is None:
        return "  n/a"
    return f"  {(new['mean']-base['mean'])/base['mean']*100:+.1f}%"

# ── SET A table ──────────────────────────────────────────────────────────

print()
print("=" * 80)
print("SET A — Heterogeneity Sweep (120 req @ 4 rps)  REPLICATION")
print("=" * 80)
POLICIES = ["rr", "lor", "po2", "mpc_po2", "kalman_po2"]
for ratio in ["1x", "2x", "4x", "8x"]:
    print(f"\n  ── {ratio} ──")
    po2 = load(f"seta_{ratio}_po2")
    for pol in POLICIES:
        s = load(f"seta_{ratio}_{pol}")
        delta = vs(po2, s) if pol not in ("rr", "po2") else ""
        print(f"  {pol:<12}{row(s)}{delta}")

# ── SET B convergence table ───────────────────────────────────────────────

print()
print("=" * 80)
print("SET B — 10K Convergence (8x hetero @ 4 rps)")
print("=" * 80)

WINDOW = 200
policies_b = ["po2", "mpc_po2", "kalman_po2"]

# Load all
data_b = {}
for pol in policies_b:
    d = load(f"setb_8x_{pol}_10k")
    if d:
        data_b[pol] = d
        print(f"  {pol}: n={d['n']}  overall mean={d['mean']:.1f}ms  p99={d['p99']:.1f}ms")

print()
print(f"  {'Window':<12}", end="")
for pol in policies_b:
    print(f"  {pol:>13}", end="")
print()
print("  " + "-" * (12 + 16 * len(policies_b)))

max_len = max((d["n"] for d in data_b.values()), default=0)
for start in range(0, max_len, WINDOW):
    print(f"  {start:5d}-{start+WINDOW:<5d} ", end="")
    base_mean = None
    for pol in policies_b:
        if pol not in data_b:
            print(f"  {'---':>13}", end=""); continue
        raw = data_b[pol]["raw"]
        ok  = [r for r in raw if r.get("ok")]
        chunk = ok[start:start+WINDOW]
        if len(chunk) < WINDOW // 2:
            print(f"  {'---':>13}", end=""); continue
        m = statistics.mean(r["latency"] for r in chunk)
        if pol == "po2":
            base_mean = m
            print(f"  {m:10.1f}ms   ", end="")
        else:
            delta = f"({(m-base_mean)/base_mean*100:+.1f}%)" if base_mean else ""
            print(f"  {m:7.1f}{delta:>6} ", end="")
    print()

# ── Kalman routing trajectory for SET B ──────────────────────────────────

print()
print("=" * 80)
print("Kalman-PO2 routing trajectory (SET B, 10K req, 8x hetero)")
print("  With 2 servers, random.sample always returns both ->")
print("  routing is deterministic. server1=slow(8 seqs), server0=fast(64 seqs)")
print("  (no server field in JSON - using latency proxy: <350ms = fast, >=350ms = slow)")
print("=" * 80)

if "kalman_po2" in data_b:
    kf_raw = [r for r in data_b["kalman_po2"]["raw"] if r.get("ok")]
    for start in range(0, len(kf_raw), 500):
        chunk = kf_raw[start:start+500]
        if not chunk: continue
        lats_c = [r["latency"] for r in chunk]
        fast_p = sum(1 for l in lats_c if l < 350) / len(lats_c)
        m = statistics.mean(lats_c)
        bar = "#" * int(fast_p * 30)
        print(f"  req {start:5d}-{start+len(chunk):<5d}: {fast_p*100:5.1f}% proxy-fast  mean={m:6.1f}ms  [{bar:<30}]")

print(f"\nFull log: {RD}/run.log")
print(f"All JSON: {RD}/")
PYEOF

echo ""
echo "Done: $(date)"
