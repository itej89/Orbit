#!/usr/bin/env python3
"""
Orbit paper benchmark script.
Sends N requests concurrently with Poisson arrivals, records latency per request.
Compares: round_robin, po2 (router without MPC), mpc_po2 (router with MPC).

Usage:
  python3 run_benchmark.py --router http://127.0.0.1:9000 \
      --model /shared_inference/models/Qwen/Qwen3-8B \
      --arrival-rate 4.0 --num-requests 100 --output results.json
"""

import argparse
import asyncio
import json
import random
import time
import statistics

import httpx

PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of World War I?",
    "Describe how neural networks learn from data.",
    "What is the difference between TCP and UDP?",
    "Summarize the plot of Hamlet.",
    "How does photosynthesis work?",
    "What are the key features of Python programming language?",
    "Explain supply and demand in economics.",
    "What is machine learning and how is it used?",
    "Describe the water cycle.",
    "How do vaccines work in the immune system?",
    "What is quantum computing?",
    "Explain the concept of recursion in programming.",
    "What are the main differences between Linux and Windows?",
    "How does the internet work?",
]


async def send_request(client, url, model, prompt, max_tokens=50):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload, timeout=120.0)
        elapsed = time.monotonic() - t0
        ok = r.status_code == 200
        return {"latency": elapsed * 1000, "ok": ok, "status": r.status_code}
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {"latency": elapsed * 1000, "ok": False, "error": str(e)}


async def run_poisson(router_url, model, arrival_rate, num_requests, max_tokens=50):
    url = f"{router_url}/v1/chat/completions"
    results = []
    tasks = []

    async with httpx.AsyncClient() as client:
        for i in range(num_requests):
            # Poisson inter-arrival: exponential gaps
            delay = random.expovariate(arrival_rate)
            await asyncio.sleep(delay)
            prompt = PROMPTS[i % len(PROMPTS)]
            task = asyncio.create_task(
                send_request(client, url, model, prompt, max_tokens)
            )
            tasks.append(task)
            print(f"  dispatched {i+1}/{num_requests}", end="\r", flush=True)

        print()
        results = await asyncio.gather(*tasks)

    return results


def summarize(results, label):
    latencies = [r["latency"] for r in results if r["ok"]]
    errors = sum(1 for r in results if not r["ok"])
    if not latencies:
        print(f"{label}: ALL FAILED")
        return {}

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    stats = {
        "label": label,
        "n_ok": len(latencies),
        "n_err": errors,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "p90_ms": latencies_sorted[int(0.90 * n)],
        "p95_ms": latencies_sorted[int(0.95 * n)],
        "p99_ms": latencies_sorted[min(int(0.99 * n), n - 1)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    return stats


def print_stats(s):
    print(f"\n  [{s['label']}]")
    print(f"    N ok/err : {s['n_ok']} / {s['n_err']}")
    print(f"    Mean     : {s['mean_ms']:.1f} ms")
    print(f"    Median   : {s['median_ms']:.1f} ms")
    print(f"    Stdev    : {s['stdev_ms']:.1f} ms")
    print(f"    P90      : {s['p90_ms']:.1f} ms")
    print(f"    P95      : {s['p95_ms']:.1f} ms")
    print(f"    P99      : {s['p99_ms']:.1f} ms")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router", default="http://127.0.0.1:9000")
    parser.add_argument("--model", default="/shared_inference/models/Qwen/Qwen3-8B")
    parser.add_argument("--arrival-rate", type=float, default=4.0,
                        help="Requests per second (Poisson rate)")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--label", default="experiment")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"\nOrbit Benchmark: {args.label}")
    print(f"  Router       : {args.router}")
    print(f"  Arrival rate : {args.arrival_rate} req/s (Poisson)")
    print(f"  Requests     : {args.num_requests}")
    print(f"  Max tokens   : {args.max_tokens}")
    print()

    results = await run_poisson(
        args.router, args.model, args.arrival_rate, args.num_requests, args.max_tokens
    )
    stats = summarize(results, args.label)
    print_stats(stats)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"stats": stats, "raw": results}, f, indent=2)
        print(f"\n  Saved → {args.output}")

    return stats


if __name__ == "__main__":
    asyncio.run(main())
