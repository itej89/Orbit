#!/bin/bash
# Orbit paper benchmark — FINAL version
# Router runs directly on host (no Docker), vLLM servers stay in Docker.
# Full provenance: all commands + results logged.
#
# Setup:
#   vllm_0: GPU 0, port 8000, max-seqs=64  (FAST)
#   vllm_1: GPU 1, port 8002, max-seqs=8   (SLOW, ~8x less capacity)
#
# Experiments:
#   1. Rate sweep: 2,4,6,8 rps — PO2 vs MPC-PO2 (find sweet spot)
#   2. Table: RR vs PO2 vs MPC-PO2 at best load (6 rps, 150 req)
#   3. Bursty: base=3 rps, burst=9 rps for 10s every 30s — PO2 vs MPC

set -e

MODEL=/shared_inference/models/Qwen/Qwen3-8B
RESULTS_BASE=/shared_inference/vpolamre/orbit/results
RUN_ID=$(date +%Y%m%d_%H%M%S)
RD="${RESULTS_BASE}/${RUN_ID}"
BENCH=/shared_inference/vpolamre/orbit/run_benchmark.py
ROUTER_PY=/shared_inference/vpolamre/orbit/orbit_router.py
LOG="${RD}/run.log"

mkdir -p "$RD"
exec > >(tee -a "$LOG") 2>&1

echo "========================================================"
echo "  Orbit Paper Benchmark — $(date)"
echo "  Run ID  : $RUN_ID"
echo "  Node    : $(hostname) ($(hostname -I | awk '{print $1}'))"
echo "  Model   : $MODEL"
echo "  Results : $RD"
echo "========================================================"
echo ""
echo "[setup] vLLM containers:"
docker ps --format "  {{.Names}}: {{.Status}}"
echo ""
echo "[setup] Verifying vLLM servers..."
for port in 8000 8002; do
    STATUS=$(curl -sf http://127.0.0.1:${port}/health && echo HEALTHY || echo DOWN)
    echo "  port ${port}: $STATUS"
done

# ---- Router management ----
ROUTER_PID=""

stop_router() {
    if [ -n "$ROUTER_PID" ] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        echo "[router] stopping PID $ROUTER_PID"
        kill "$ROUTER_PID" 2>/dev/null || true
        sleep 2
    fi
    ROUTER_PID=""
    # Kill any leftover uvicorn on 9000
    fuser -k 9000/tcp 2>/dev/null || true
    sleep 1
}

start_router() {
    local policy=$1
    local enable_mpc=$2  # "" or "--enable-mpc"
    stop_router
    echo ""
    CMD="python3 ${ROUTER_PY} --port 9000 --policy ${policy} ${enable_mpc} single --servers 127.0.0.1:8000 127.0.0.1:8002"
    echo "[router] CMD: $CMD"
    $CMD &
    ROUTER_PID=$!
    echo "[router] PID=$ROUTER_PID, waiting for /health..."
    for i in $(seq 1 20); do
        if curl -sf http://127.0.0.1:9000/health > /dev/null 2>&1; then
            echo "[router] healthy after ${i}x1s"
            return 0
        fi
        sleep 1
    done
    echo "[router] FAILED to become healthy"
    exit 1
}

run_bench() {
    local label=$1
    local output=$2
    local arrival=$3
    local nreqs=$4
    local maxtok=${5:-50}
    echo ""
    CMD="python3 ${BENCH} --router http://127.0.0.1:9000 --model ${MODEL} --arrival-rate ${arrival} --num-requests ${nreqs} --max-tokens ${maxtok} --label ${label} --output ${output}"
    echo "[bench] CMD: $CMD"
    $CMD
}

dump_metrics() {
    local tag=$1
    echo ""
    echo "[metrics:${tag}]"
    curl -s http://127.0.0.1:9000/metrics | python3 -m json.tool
}

trap stop_router EXIT

# ================================================================
# SET 1: Arrival rate sweep (PO2 vs MPC-PO2)
# ================================================================
echo ""
echo "================================================================"
echo "SET 1: Rate sweep — PO2 vs MPC-PO2 (100 req each)"
echo "================================================================"

for RATE in 2.0 4.0 6.0 8.0; do
    echo ""
    echo "---- Rate = ${RATE} rps ----"

    start_router "po2" ""
    sleep 2
    run_bench "po2_${RATE}rps" "${RD}/po2_${RATE}rps.json" "$RATE" 100 50
    dump_metrics "po2_${RATE}rps"

    start_router "mpc_po2" "--enable-mpc"
    sleep 5   # MPC settles quickly since no load yet
    run_bench "mpc_${RATE}rps" "${RD}/mpc_${RATE}rps.json" "$RATE" 100 50
    dump_metrics "mpc_${RATE}rps"
done

# ================================================================
# SET 2: Main comparison table — RR vs PO2 vs MPC at 6 rps
# ================================================================
echo ""
echo "================================================================"
echo "SET 2: RR vs PO2 vs MPC-PO2 at 6 rps (150 requests) — Table 2"
echo "================================================================"

start_router "rr" ""
sleep 2
run_bench "table_rr" "${RD}/table_rr.json" 6.0 150 50
dump_metrics "table_rr"

start_router "po2" ""
sleep 2
run_bench "table_po2" "${RD}/table_po2.json" 6.0 150 50
dump_metrics "table_po2"

start_router "mpc_po2" "--enable-mpc"
sleep 5
run_bench "table_mpc" "${RD}/table_mpc.json" 6.0 150 50
dump_metrics "table_mpc"

# ================================================================
# SET 3: Bursty workload — Table 3
# ================================================================
echo ""
echo "================================================================"
echo "SET 3: Bursty — base=3rps, burst=9rps for 10s every 30s (200 req)"
echo "================================================================"

cat > /tmp/orbit_bursty.py << 'PYEOF'
#!/usr/bin/env python3
import asyncio, json, random, time, statistics, httpx, argparse

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of World War I?",
    "Describe how neural networks learn from data.",
    "What is the difference between TCP and UDP?",
    "Summarize the plot of Hamlet.",
    "How does photosynthesis work?",
    "What is machine learning and how is it used?",
    "How does the internet work?",
    "What is quantum computing?",
    "Explain the concept of recursion in programming.",
]

async def send_req(client, url, model, prompt, max_tokens=50):
    payload = {"model": model, "messages": [{"role":"user","content":prompt}],
               "max_tokens": max_tokens, "stream": False}
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload, timeout=120.0)
        return {"latency": (time.monotonic()-t0)*1000, "ok": r.status_code==200}
    except Exception as e:
        return {"latency": (time.monotonic()-t0)*1000, "ok": False, "error": str(e)}

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--router", default="http://127.0.0.1:9000")
    p.add_argument("--model", default="/shared_inference/models/Qwen/Qwen3-8B")
    p.add_argument("--base-rate", type=float, default=3.0)
    p.add_argument("--burst-mult", type=float, default=3.0)
    p.add_argument("--burst-period", type=float, default=30.0)
    p.add_argument("--burst-dur", type=float, default=10.0)
    p.add_argument("--num-requests", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--label", default="bursty")
    p.add_argument("--output")
    args = p.parse_args()

    url = f"{args.router}/v1/chat/completions"
    tasks = []
    t0 = time.monotonic()

    print(f"\nBursty Benchmark: {args.label}")
    print(f"  base={args.base_rate}rps  burst={args.burst_mult}x/{args.burst_dur}s every {args.burst_period}s")

    async with httpx.AsyncClient() as client:
        for i in range(args.num_requests):
            phase = (time.monotonic()-t0) % args.burst_period
            rate = args.base_rate * args.burst_mult if phase < args.burst_dur else args.base_rate
            await asyncio.sleep(random.expovariate(rate))
            tasks.append(asyncio.create_task(
                send_req(client, url, args.model, PROMPTS[i % len(PROMPTS)], args.max_tokens)))
            print(f"  dispatched {i+1}/{args.num_requests} @{rate:.0f}rps", end="\r", flush=True)
        print()
        results = await asyncio.gather(*tasks)

    lats = sorted(r["latency"] for r in results if r["ok"])
    n = len(lats)
    errs = sum(1 for r in results if not r["ok"])
    if n == 0:
        print("ALL FAILED"); return
    s = {
        "label": args.label, "n_ok": n, "n_err": errs,
        "mean_ms": statistics.mean(lats), "median_ms": statistics.median(lats),
        "stdev_ms": statistics.stdev(lats) if n>1 else 0,
        "p90_ms": lats[int(0.90*n)], "p95_ms": lats[int(0.95*n)],
        "p99_ms": lats[min(int(0.99*n),n-1)],
    }
    print(f"\n  [{args.label}]")
    for k,v in s.items():
        print(f"    {k}: {v:.1f}" if isinstance(v,float) else f"    {k}: {v}")
    if args.output:
        with open(args.output,"w") as f: json.dump({"stats":s,"raw":results},f,indent=2)
        print(f"  Saved → {args.output}")

asyncio.run(main())
PYEOF

start_router "po2" ""
sleep 2
echo "[bench] CMD: python3 /tmp/orbit_bursty.py --router http://127.0.0.1:9000 --model ${MODEL} --base-rate 3.0 --burst-mult 3.0 --burst-period 30 --burst-dur 10 --num-requests 200 --label bursty_po2 --output ${RD}/bursty_po2.json"
python3 /tmp/orbit_bursty.py \
    --router "http://127.0.0.1:9000" --model "$MODEL" \
    --base-rate 3.0 --burst-mult 3.0 --burst-period 30 --burst-dur 10 \
    --num-requests 200 --label bursty_po2 --output "${RD}/bursty_po2.json"
dump_metrics "bursty_po2"

start_router "mpc_po2" "--enable-mpc"
sleep 5
echo "[bench] CMD: python3 /tmp/orbit_bursty.py --router http://127.0.0.1:9000 --model ${MODEL} --base-rate 3.0 --burst-mult 3.0 --burst-period 30 --burst-dur 10 --num-requests 200 --label bursty_mpc --output ${RD}/bursty_mpc.json"
python3 /tmp/orbit_bursty.py \
    --router "http://127.0.0.1:9000" --model "$MODEL" \
    --base-rate 3.0 --burst-mult 3.0 --burst-period 30 --burst-dur 10 \
    --num-requests 200 --label bursty_mpc --output "${RD}/bursty_mpc.json"
dump_metrics "bursty_mpc"

stop_router

# ================================================================
# FINAL SUMMARY TABLE
# ================================================================
echo ""
echo "================================================================"
echo "FINAL RESULTS SUMMARY — $(date)"
echo "================================================================"

python3 << PYEOF
import json, os

RD = "${RD}"

def load(f):
    p = f"{RD}/{f}.json"
    if not os.path.exists(p): return None
    return json.load(open(p))["stats"]

def pct(base, new, key):
    if base is None or new is None: return "n/a"
    return f"{(base[key]-new[key])/base[key]*100:+.1f}%"

print("\n--- SET 1: Rate sweep ---")
print(f"{'Rate':>6}  {'Policy':<14}  {'Mean(ms)':>9}  {'Stdev(ms)':>10}  {'P99(ms)':>9}  {'N':>5}")
print("-" * 58)
for rate in [2.0, 4.0, 6.0, 8.0]:
    for tag in ["po2", "mpc"]:
        fn = f"po2_{rate}rps" if tag=="po2" else f"mpc_{rate}rps"
        s = load(fn)
        if s: print(f"{rate:>6.1f}  {s['label']:<14}  {s['mean_ms']:>9.1f}  {s['stdev_ms']:>10.1f}  {s['p99_ms']:>9.1f}  {s['n_ok']:>5}")

print("\n--- SET 2: Main comparison table (6 rps, 150 req) ---")
print(f"{'Policy':<18}  {'Mean(ms)':>9}  {'Stdev(ms)':>10}  {'P95(ms)':>9}  {'P99(ms)':>9}  {'N':>5}")
print("-" * 66)
rr  = load("table_rr")
po2 = load("table_po2")
mpc = load("table_mpc")
for s in [rr, po2, mpc]:
    if s: print(f"{s['label']:<18}  {s['mean_ms']:>9.1f}  {s['stdev_ms']:>10.1f}  {s['p95_ms']:>9.1f}  {s['p99_ms']:>9.1f}  {s['n_ok']:>5}")
if po2 and mpc:
    print(f"\nOrbit MPC-PO2 vs PO2:  mean={pct(po2,mpc,'mean_ms')}  P99={pct(po2,mpc,'p99_ms')}  stdev={pct(po2,mpc,'stdev_ms')}")

print("\n--- SET 3: Bursty workload (base=3rps, burst=9rps) ---")
print(f"{'Policy':<18}  {'Mean(ms)':>9}  {'Stdev(ms)':>10}  {'P99(ms)':>9}  {'N':>5}")
print("-" * 55)
bpo2 = load("bursty_po2")
bmpc = load("bursty_mpc")
for s in [bpo2, bmpc]:
    if s: print(f"{s['label']:<18}  {s['mean_ms']:>9.1f}  {s['stdev_ms']:>10.1f}  {s['p99_ms']:>9.1f}  {s['n_ok']:>5}")
if bpo2 and bmpc:
    print(f"\nOrbit MPC-PO2 vs PO2:  P99={pct(bpo2,bmpc,'p99_ms')}  stdev={pct(bpo2,bmpc,'stdev_ms')}")

print(f"\nRaw data + full log: {RD}/")
PYEOF

echo ""
echo "Done: $(date)"
