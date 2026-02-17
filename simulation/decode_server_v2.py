#!/usr/bin/env python3
"""
Enhanced Decode Server Simulation

Simulates realistic decode behavior with:
- Variable token delay
- Generation length variability
- Capacity simulation (TP-like behavior)
- Load-dependent slowdown
- Batch effects simulation
"""

import argparse
import asyncio
import random
import time
import json
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

# Metrics tracking
request_times = deque(maxlen=1000)
token_times = deque(maxlen=5000)
active_requests = 0


@app.post("/v1/chat/completions")
async def decode(req: Request):
    global active_requests
    
    body = await req.json()
    
    # Get max tokens (variable generation length)
    max_tokens = body.get("max_tokens", 50)
    
    # Determine actual generation length (geometric distribution)
    if app.state.variable_length:
        # Mean around max_tokens/2, bounded by max_tokens
        gen_length = min(max_tokens, int(random.expovariate(1.0 / (max_tokens / 2))) + 1)
    else:
        gen_length = max_tokens
    
    active_requests += 1
    start_time = time.monotonic()
    active_count_local = active_requests  # Capture for use in generator
    
    async def generate_tokens():
        global active_requests
        
        try:
            for i in range(gen_length):
                # Base token delay
                token_delay = app.state.base_token_delay
                
                # Add variance
                if app.state.variance > 0:
                    token_delay *= random.uniform(1 - app.state.variance, 1 + app.state.variance)
                
                # Load-dependent slowdown (simulates memory bandwidth contention)
                if app.state.load_factor > 0:
                    load_multiplier = 1 + (active_requests * app.state.load_factor)
                    token_delay *= load_multiplier
                
                # Batch effect: first token slower (prefill residual)
                if i == 0:
                    token_delay *= 2.0
                
                # Capacity slowdown
                if app.state.capacity > 0 and active_requests > app.state.capacity:
                    token_delay *= (1 + 0.1 * (active_requests - app.state.capacity))
                
                token_start = time.monotonic()
                await asyncio.sleep(token_delay)
                token_times.append(time.monotonic() - token_start)
                
                # Yield token
                yield json.dumps({
                    "choices": [{
                        "delta": {"content": f"token_{i}"},
                        "index": 0
                    }],
                    "token_index": i,
                }).encode() + b"\n"
            
            # Final message
            elapsed = time.monotonic() - start_time
            request_times.append(elapsed)
            
            yield json.dumps({
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0
                }],
                "usage": {
                    "completion_tokens": gen_length,
                    "total_time_ms": elapsed * 1000,
                }
            }).encode() + b"\n"
            
        finally:
            active_requests -= 1
    
    return StreamingResponse(generate_tokens(), media_type="application/x-ndjson")


@app.get("/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/metrics")
async def metrics():
    result = {
        "active_requests": active_requests,
        "capacity": app.state.capacity,
        "base_token_delay_ms": app.state.base_token_delay * 1000,
    }
    
    if request_times:
        times = list(request_times)
        result.update({
            "total_requests": len(times),
            "mean_latency_ms": sum(times) / len(times) * 1000,
            "p99_latency_ms": sorted(times)[int(len(times) * 0.99)] * 1000 if len(times) > 10 else 0,
        })
    
    if token_times:
        ttimes = list(token_times)
        result.update({
            "mean_itl_ms": sum(ttimes) / len(ttimes) * 1000,
            "tokens_generated": len(ttimes),
        })
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Enhanced Decode Server Simulation")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--base-token-delay", type=float, default=0.010,
                        help="Base delay per token in seconds (default: 10ms)")
    parser.add_argument("--variance", type=float, default=0.2,
                        help="Token delay variance 0-1 (default: 0.2 = ±20%%)")
    parser.add_argument("--capacity", type=int, default=8,
                        help="Max concurrent requests before slowdown (default: 8)")
    parser.add_argument("--load-factor", type=float, default=0.03,
                        help="Load-dependent slowdown factor (default: 0.03)")
    parser.add_argument("--variable-length", action="store_true", default=True,
                        help="Use variable generation lengths (default: True)")
    parser.add_argument("--fixed-length", action="store_false", dest="variable_length",
                        help="Use fixed generation lengths")
    
    # Preset configurations
    parser.add_argument("--preset", type=str, choices=["fast", "medium", "slow", "variable"],
                        help="Use a preset configuration")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == "fast":
        args.base_token_delay = 0.008
        args.capacity = 16
        args.variance = 0.1
    elif args.preset == "medium":
        args.base_token_delay = 0.015
        args.capacity = 8
        args.variance = 0.2
    elif args.preset == "slow":
        args.base_token_delay = 0.040
        args.capacity = 4
        args.variance = 0.3
    elif args.preset == "variable":
        args.base_token_delay = 0.020
        args.capacity = 8
        args.variance = 0.5
    
    app.state.base_token_delay = args.base_token_delay
    app.state.variance = args.variance
    app.state.capacity = args.capacity
    app.state.load_factor = args.load_factor
    app.state.variable_length = args.variable_length
    
    print(f"Starting Decode Server on port {args.port}")
    print(f"  Base token delay: {args.base_token_delay*1000:.1f}ms")
    print(f"  Variance: ±{args.variance*100:.0f}%")
    print(f"  Capacity: {args.capacity}")
    print(f"  Load factor: {args.load_factor}")
    print(f"  Variable length: {args.variable_length}")
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
