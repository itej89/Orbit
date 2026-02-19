#!/usr/bin/env python3
"""
Enhanced Benchmark Script for Orbit Experiments

Features:
- Multiple workload patterns (steady, bursty, ramping)
- Configurable request parameters
- Real-time metrics collection
- Comprehensive result analysis
"""

import asyncio
import httpx
import time
import random
import math
import csv
import json
import signal
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Configuration
@dataclass
class BenchmarkConfig:
    url: str = "http://127.0.0.1:8000/v1/chat/completions"
    metrics_url: str = "http://127.0.0.1:8000/metrics"
    concurrency: int = 32
    total_requests: int = 500
    target_rps: float = 20.0
    duration_seconds: float = 0  # If > 0, run for this duration instead of total_requests
    
    # Workload pattern
    pattern: str = "steady"  # steady, bursty, ramping
    burst_multiplier: float = 3.0
    burst_duration: float = 2.0
    burst_interval: float = 10.0
    
    # Request parameters
    min_prompt_tokens: int = 50
    max_prompt_tokens: int = 500
    min_output_tokens: int = 16
    max_output_tokens: int = 128
    
    # Output
    output_dir: str = "results"
    experiment_name: str = "experiment"


# Global state
stop_event = asyncio.Event()
request_records = asyncio.Queue()
metrics_records = asyncio.Queue()


@dataclass
class RequestRecord:
    timestamp: float
    worker_id: int
    latency_ms: float
    ttft_ms: float  # Time to first token
    tokens_generated: int
    status: int
    prompt_tokens: int


def generate_prompt(min_tokens: int, max_tokens: int) -> str:
    """Generate a prompt with approximately the specified token count."""
    target_tokens = random.randint(min_tokens, max_tokens)
    words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
             "AI", "machine", "learning", "neural", "network", "transformer", "model",
             "inference", "training", "optimization", "performance", "latency", "throughput"]
    prompt_words = [random.choice(words) for _ in range(int(target_tokens * 0.75))]
    return " ".join(prompt_words)


def get_arrival_rate(config: BenchmarkConfig, elapsed: float) -> float:
    """Get current arrival rate based on workload pattern."""
    base_rate = config.target_rps
    
    if config.pattern == "steady":
        return base_rate
    
    elif config.pattern == "bursty":
        # Periodic bursts
        cycle_pos = elapsed % config.burst_interval
        if cycle_pos < config.burst_duration:
            return base_rate * config.burst_multiplier
        return base_rate
    
    elif config.pattern == "ramping":
        # Gradual ramp up then down
        if elapsed < 30:
            return base_rate * (0.2 + 0.8 * elapsed / 30)
        elif elapsed < 60:
            return base_rate
        else:
            return base_rate * max(0.2, 1 - 0.8 * (elapsed - 60) / 30)
    
    return base_rate


async def send_request(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    worker_id: int,
    request_num: int
) -> None:
    """Send a single request and record metrics."""
    
    # Generate request
    prompt = generate_prompt(config.min_prompt_tokens, config.max_prompt_tokens)
    max_tokens = random.randint(config.min_output_tokens, config.max_output_tokens)
    
    request_body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    
    prompt_tokens = len(prompt.split())
    start_time = time.monotonic()
    ttft = None
    tokens = 0
    status = -1
    
    try:
        async with client.stream("POST", config.url, json=request_body) as resp:
            status = resp.status_code
            async for line in resp.aiter_lines():
                if line.strip():
                    if ttft is None:
                        ttft = (time.monotonic() - start_time) * 1000
                    tokens += 1
    except Exception as e:
        pass
    
    latency_ms = (time.monotonic() - start_time) * 1000
    
    record = RequestRecord(
        timestamp=time.time(),
        worker_id=worker_id,
        latency_ms=latency_ms,
        ttft_ms=ttft or latency_ms,
        tokens_generated=tokens,
        status=status,
        prompt_tokens=prompt_tokens,
    )
    await request_records.put(record)


async def worker(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    worker_id: int,
    start_time: float,
    request_counter: List[int],
    counter_lock: asyncio.Lock,
) -> None:
    """Worker that sends requests according to arrival pattern."""
    
    while not stop_event.is_set():
        # Check if we've hit request limit
        async with counter_lock:
            if config.total_requests > 0 and request_counter[0] >= config.total_requests:
                break
            request_num = request_counter[0]
            request_counter[0] += 1
        
        # Check duration limit
        elapsed = time.monotonic() - start_time
        if config.duration_seconds > 0 and elapsed >= config.duration_seconds:
            break
        
        # Get current arrival rate and compute sleep
        rate = get_arrival_rate(config, elapsed)
        if rate > 0:
            # Poisson inter-arrival
            sleep_time = random.expovariate(rate / config.concurrency)
            await asyncio.sleep(sleep_time)
        
        # Send request
        await send_request(client, config, worker_id, request_num)


async def metrics_collector(client: httpx.AsyncClient, config: BenchmarkConfig) -> None:
    """Periodically collect metrics from the router."""
    while not stop_event.is_set():
        try:
            resp = await client.get(config.metrics_url)
            if resp.status_code == 200:
                data = resp.json()
                data["timestamp"] = time.time()
                await metrics_records.put(data)
        except Exception:
            pass
        await asyncio.sleep(1.0)


async def results_writer(output_dir: Path) -> None:
    """Write results to CSV files as they arrive."""
    requests_file = output_dir / "requests.csv"
    
    fieldnames = ["timestamp", "worker_id", "latency_ms", "ttft_ms", 
                  "tokens_generated", "status", "prompt_tokens"]
    
    with open(requests_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        while True:
            try:
                record = await asyncio.wait_for(request_records.get(), timeout=0.5)
                writer.writerow(asdict(record))
                f.flush()
                request_records.task_done()
            except asyncio.TimeoutError:
                if stop_event.is_set() and request_records.empty():
                    break


async def metrics_writer(output_dir: Path) -> None:
    """Write metrics to JSON file."""
    metrics_file = output_dir / "metrics.jsonl"
    
    with open(metrics_file, "w") as f:
        while True:
            try:
                record = await asyncio.wait_for(metrics_records.get(), timeout=0.5)
                f.write(json.dumps(record) + "\n")
                f.flush()
                metrics_records.task_done()
            except asyncio.TimeoutError:
                if stop_event.is_set() and metrics_records.empty():
                    break


def compute_summary(output_dir: Path) -> Dict:
    """Compute summary statistics from results."""
    requests_file = output_dir / "requests.csv"
    
    latencies = []
    ttfts = []
    tokens = []
    successes = 0
    total = 0
    
    with open(requests_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if int(row["status"]) == 200:
                successes += 1
                latencies.append(float(row["latency_ms"]))
                ttfts.append(float(row["ttft_ms"]))
                tokens.append(int(row["tokens_generated"]))
    
    if not latencies:
        return {"error": "No successful requests"}
    
    latencies.sort()
    ttfts.sort()
    
    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]
    
    summary = {
        "total_requests": total,
        "successful_requests": successes,
        "success_rate": successes / total * 100 if total > 0 else 0,
        "latency": {
            "mean_ms": sum(latencies) / len(latencies),
            "p50_ms": percentile(latencies, 50),
            "p90_ms": percentile(latencies, 90),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "std_ms": (sum((x - sum(latencies)/len(latencies))**2 for x in latencies) / len(latencies)) ** 0.5,
        },
        "ttft": {
            "mean_ms": sum(ttfts) / len(ttfts),
            "p50_ms": percentile(ttfts, 50),
            "p99_ms": percentile(ttfts, 99),
        },
        "tokens": {
            "mean": sum(tokens) / len(tokens),
            "total": sum(tokens),
        },
    }
    
    return summary


async def run_benchmark(config: BenchmarkConfig) -> Dict:
    """Run the benchmark."""
    output_dir = Path(config.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Running benchmark: {config.experiment_name}")
    print(f"{'='*60}")
    print(f"  URL: {config.url}")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Total requests: {config.total_requests}")
    print(f"  Target RPS: {config.target_rps}")
    print(f"  Pattern: {config.pattern}")
    print(f"  Output: {output_dir}")
    print()
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        # Start writers
        results_task = asyncio.create_task(results_writer(output_dir))
        metrics_task = asyncio.create_task(metrics_writer(output_dir))
        metrics_collector_task = asyncio.create_task(metrics_collector(client, config))
        
        # Start workers
        start_time = time.monotonic()
        request_counter = [0]
        counter_lock = asyncio.Lock()
        
        workers = [
            asyncio.create_task(worker(client, config, i, start_time, request_counter, counter_lock))
            for i in range(config.concurrency)
        ]
        
        # Progress reporting
        last_count = 0
        while not all(w.done() for w in workers):
            await asyncio.sleep(2)
            current = request_counter[0]
            elapsed = time.monotonic() - start_time
            rps = (current - last_count) / 2
            print(f"  Progress: {current}/{config.total_requests} requests, "
                  f"{rps:.1f} req/s, {elapsed:.1f}s elapsed")
            last_count = current
        
        await asyncio.gather(*workers)
        
        # Stop collectors
        stop_event.set()
        await metrics_collector_task
        
        # Drain queues
        await request_records.join()
        await metrics_records.join()
        await results_task
        await metrics_task
    
    # Compute summary
    summary = compute_summary(output_dir)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  Total requests: {summary.get('total_requests', 'N/A')}")
    print(f"  Success rate: {summary.get('success_rate', 0):.1f}%")
    print(f"  Mean latency: {summary.get('latency', {}).get('mean_ms', 0):.1f}ms")
    print(f"  P99 latency: {summary.get('latency', {}).get('p99_ms', 0):.1f}ms")
    print(f"  Std dev: {summary.get('latency', {}).get('std_ms', 0):.1f}ms")
    print(f"\nResults saved to: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Orbit Benchmark")
    
    # Basic config
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=500, dest="total_requests")
    parser.add_argument("--rps", type=float, default=20.0, dest="target_rps")
    parser.add_argument("--duration", type=float, default=0, dest="duration_seconds")
    
    # Workload
    parser.add_argument("--pattern", choices=["steady", "bursty", "ramping"], default="steady")
    parser.add_argument("--burst-multiplier", type=float, default=3.0)
    
    # Prompt config (ISL = Input Sequence Length, OSL = Output Sequence Length)
    parser.add_argument("--min-prompt", type=int, default=50, dest="min_prompt_tokens")
    parser.add_argument("--max-prompt", type=int, default=500, dest="max_prompt_tokens")
    parser.add_argument("--min-output", type=int, default=16, dest="min_output_tokens")
    parser.add_argument("--max-output", type=int, default=128, dest="max_output_tokens")
    # Aliases for ISL/OSL
    parser.add_argument("--isl-min", type=int, dest="min_prompt_tokens")
    parser.add_argument("--isl-max", type=int, dest="max_prompt_tokens")
    parser.add_argument("--osl-min", type=int, dest="min_output_tokens")
    parser.add_argument("--osl-max", type=int, dest="max_output_tokens")
    
    # Output
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default="experiment", dest="experiment_name")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        url=args.url,
        metrics_url=args.metrics_url,
        concurrency=args.concurrency,
        total_requests=args.total_requests,
        target_rps=args.target_rps,
        duration_seconds=args.duration_seconds,
        pattern=args.pattern,
        burst_multiplier=args.burst_multiplier,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\nStopping benchmark...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
