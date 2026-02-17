#!/usr/bin/env python3
"""
Enhanced Prefill Server Simulation

Simulates realistic prefill behavior with:
- Variable delay based on prompt length
- Configurable base latency and variance
- Server capacity simulation (TP-like behavior)
- Load-dependent slowdown
"""

import argparse
import asyncio
import random
import time
import math
from collections import deque
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# Metrics tracking
request_times = deque(maxlen=1000)
active_requests = 0


@app.post("/v1/chat/completions")
async def prefill(req: Request):
    global active_requests
    
    body = await req.json()
    
    # Extract prompt info
    messages = body.get("messages", [])
    prompt_text = " ".join([m.get("content", "") for m in messages])
    prompt_tokens = len(prompt_text.split()) * 1.3  # Rough token estimate
    
    # Base delay calculation
    base_delay = app.state.base_delay
    delay_per_token = app.state.delay_per_token
    
    # Compute prefill time based on prompt length
    prefill_time = base_delay + (prompt_tokens * delay_per_token)
    
    # Add variance (simulate real-world jitter)
    variance = app.state.variance
    if variance > 0:
        prefill_time *= random.uniform(1 - variance, 1 + variance)
    
    # Load-dependent slowdown (simulates GPU contention)
    if app.state.load_factor > 0:
        load_multiplier = 1 + (active_requests * app.state.load_factor)
        prefill_time *= load_multiplier
    
    # Capacity limit simulation (simulate TP effect)
    capacity = app.state.capacity
    if capacity > 0 and active_requests >= capacity:
        # Queue delay when over capacity
        queue_wait = (active_requests - capacity + 1) * 0.01
        prefill_time += queue_wait
    
    # Track metrics
    active_requests += 1
    start_time = time.monotonic()
    
    try:
        await asyncio.sleep(prefill_time)
        
        elapsed = time.monotonic() - start_time
        request_times.append(elapsed)
        
        return {
            "kv_transfer_params": {
                "dummy": True,
                "prefill_time_ms": elapsed * 1000,
                "prompt_tokens": int(prompt_tokens),
            }
        }
    finally:
        active_requests -= 1


@app.get("/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/metrics")
async def metrics():
    if request_times:
        times = list(request_times)
        return {
            "active_requests": active_requests,
            "total_requests": len(times),
            "mean_latency_ms": sum(times) / len(times) * 1000,
            "p99_latency_ms": sorted(times)[int(len(times) * 0.99)] * 1000 if len(times) > 10 else 0,
            "capacity": app.state.capacity,
            "base_delay_ms": app.state.base_delay * 1000,
        }
    return {"active_requests": active_requests, "total_requests": 0}


def main():
    parser = argparse.ArgumentParser(description="Enhanced Prefill Server Simulation")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--base-delay", type=float, default=0.005, 
                        help="Base delay in seconds (default: 5ms)")
    parser.add_argument("--delay-per-token", type=float, default=0.00001,
                        help="Additional delay per token (default: 0.01ms)")
    parser.add_argument("--variance", type=float, default=0.2,
                        help="Delay variance factor 0-1 (default: 0.2 = ±20%%)")
    parser.add_argument("--capacity", type=int, default=8,
                        help="Max concurrent requests before queuing (default: 8)")
    parser.add_argument("--load-factor", type=float, default=0.02,
                        help="Load-dependent slowdown factor (default: 0.02)")
    
    # Preset configurations for easy testing
    parser.add_argument("--preset", type=str, choices=["fast", "medium", "slow", "variable"],
                        help="Use a preset configuration")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == "fast":
        args.base_delay = 0.005
        args.capacity = 16
        args.variance = 0.1
    elif args.preset == "medium":
        args.base_delay = 0.020
        args.capacity = 8
        args.variance = 0.2
    elif args.preset == "slow":
        args.base_delay = 0.050
        args.capacity = 4
        args.variance = 0.3
    elif args.preset == "variable":
        args.base_delay = 0.015
        args.capacity = 8
        args.variance = 0.5
    
    app.state.base_delay = args.base_delay
    app.state.delay_per_token = args.delay_per_token
    app.state.variance = args.variance
    app.state.capacity = args.capacity
    app.state.load_factor = args.load_factor
    
    print(f"Starting Prefill Server on port {args.port}")
    print(f"  Base delay: {args.base_delay*1000:.1f}ms")
    print(f"  Delay/token: {args.delay_per_token*1000:.3f}ms")
    print(f"  Variance: ±{args.variance*100:.0f}%")
    print(f"  Capacity: {args.capacity}")
    print(f"  Load factor: {args.load_factor}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
