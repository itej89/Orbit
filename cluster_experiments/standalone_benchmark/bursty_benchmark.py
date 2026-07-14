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
