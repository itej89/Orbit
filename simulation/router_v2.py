#!/usr/bin/env python3
"""
Enhanced Orbit Router with Configurable MPC

Supports:
- Multiple routing policies: rr, random, po2, weighted_po2
- MPC enable/disable via flag
- Configurable MPC parameters
- Comprehensive metrics
"""

import argparse
import itertools
import logging
import time
import uuid
import asyncio
import random
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

# Optional: cvxpy for MPC (graceful fallback if not available)
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("WARNING: cvxpy not available, MPC disabled")

logger = logging.getLogger(__name__)

# Global args
global_args = None

# -----------------------------
# Configuration
# -----------------------------
MPC_DT = 0.1          # MPC control interval (seconds)
RATE_WINDOW = 0.3     # Rate estimation window

# -----------------------------
# Utilities
# -----------------------------
class RateEstimator:
    """Rate estimator using sliding window + EWMA."""
    def __init__(self, window_sec=0.5, alpha=0.3):
        self.window = window_sec
        self.alpha = alpha
        self.count = 0
        self.last = time.monotonic()
        self.rate = 0.0

    def tick(self, n=1):
        now = time.monotonic()
        self.count += n
        elapsed = now - self.last
        if elapsed >= self.window:
            instant_rate = self.count / elapsed
            self.rate = self.alpha * instant_rate + (1 - self.alpha) * self.rate
            self.count = 0
            self.last = now

    def get(self):
        return self.rate


class EWMAEstimator:
    """Exponential weighted moving average."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value

    def get(self):
        return 0.0 if self.value is None else float(self.value)


# -----------------------------
# MPC Controller
# -----------------------------
class MPCController:
    def __init__(self, horizon=10, dt=0.1, lambda_reg=0.5, min_w=0.1, max_w=3.0):
        self.horizon = horizon
        self.dt = dt
        self.lambda_reg = lambda_reg
        self.min_w = min_w
        self.max_w = max_w
        self.weights = {}
        self.enabled = HAS_CVXPY
        
    async def control_loop(self, app):
        """Background MPC control loop."""
        if not self.enabled:
            return
            
        while True:
            try:
                await self._update_weights(app)
            except Exception as e:
                logger.warning(f"MPC error: {e}")
            await asyncio.sleep(self.dt)
    
    async def _update_weights(self, app):
        inflight = app.state.metrics["decode_inflight"]
        arrival_rate = app.state.metrics["arrival_rate"].get()
        service_rates = app.state.metrics["service_rate"]
        
        if not inflight:
            return
        
        # Normalize queues
        avg_q = max(1.0, sum(inflight.values()) / len(inflight))
        
        eps = 1e-6
        total_weight = max(sum(self.weights.values()), eps)
        
        for wid, q0 in inflight.items():
            sr = service_rates.get(wid)
            if sr is None:
                continue
            mu = sr.get()
            if mu < eps:
                mu = 1.0
            
            q_scaled = q0 / avg_q
            target_q = 1.0
            
            # Solve QP
            try:
                w = cp.Variable(self.horizon)
                q = cp.Variable(self.horizon + 1)
                
                constraints = [q[0] == q_scaled]
                cost = 0
                
                for k in range(self.horizon):
                    p_i = w[k] / (total_weight + eps)
                    constraints += [
                        q[k + 1] == q[k] + self.dt * (arrival_rate * p_i - mu) / max(avg_q, eps),
                        q[k + 1] >= 0,
                        w[k] >= self.min_w,
                        w[k] <= self.max_w,
                    ]
                    cost += cp.square(q[k + 1] - target_q) + self.lambda_reg * cp.square(w[k] - 1.0)
                
                prob = cp.Problem(cp.Minimize(cost), constraints)
                prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                
                if w.value is not None:
                    w_new = float(w.value[0])
                    w_old = self.weights.get(wid, 1.0)
                    # Smooth update
                    w_smooth = 0.25 * w_new + 0.75 * w_old
                    self.weights[wid] = max(self.min_w, min(self.max_w, w_smooth))
            except Exception as e:
                pass  # Keep existing weight
    
    def get_weight(self, wid) -> float:
        return self.weights.get(wid, 1.0)


# -----------------------------
# Routing Policies
# -----------------------------
def select_random(pool, inflight, mpc_weights):
    """Random selection."""
    return random.choice(pool)


def select_round_robin(pool, inflight, mpc_weights, rr_iter):
    """Round robin selection."""
    return pool[next(rr_iter) % len(pool)]


def select_po2_vanilla(pool, inflight, mpc_weights):
    """Power-of-Two-Choices without MPC weights."""
    if len(pool) == 1:
        return pool[0]
    
    a, b = random.sample(pool, 2)
    score_a = inflight.get(a["id"], 0)
    score_b = inflight.get(b["id"], 0)
    return a if score_a <= score_b else b


def select_po2_weighted(pool, inflight, mpc_weights):
    """Power-of-Two-Choices with MPC weights."""
    if len(pool) == 1:
        return pool[0]
    
    a, b = random.sample(pool, 2)
    
    wa = mpc_weights.get(a["id"], 1.0)
    wb = mpc_weights.get(b["id"], 1.0)
    
    score_a = inflight.get(a["id"], 0) / wa
    score_b = inflight.get(b["id"], 0) / wb
    return a if score_a <= score_b else b


def select_weighted_rr(pool, inflight, mpc_weights, rr_iter):
    """Weighted round robin with MPC weights."""
    for _ in range(len(pool)):
        idx = next(rr_iter) % len(pool)
        worker = pool[idx]
        weight = mpc_weights.get(worker["id"], 1.0)
        # Probabilistic skip based on weight
        if random.random() <= min(weight, 1.0):
            return worker
    return pool[idx]


# -----------------------------
# FastAPI App
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    args = global_args
    
    # Initialize clients
    app.state.prefill_clients = []
    app.state.decode_clients = []
    
    for i, (host, port) in enumerate(args.prefiller_instances):
        app.state.prefill_clients.append({
            "client": httpx.AsyncClient(timeout=httpx.Timeout(60.0)),
            "host": host, "port": port, "id": i
        })
    
    for i, (host, port) in enumerate(args.decoder_instances):
        app.state.decode_clients.append({
            "client": httpx.AsyncClient(timeout=httpx.Timeout(60.0)),
            "host": host, "port": port, "id": i
        })
    
    # Initialize metrics
    app.state.metrics = {
        "arrival_rate": RateEstimator(window_sec=RATE_WINDOW),
        "decode_latency": EWMAEstimator(),
        "prefill_latency": EWMAEstimator(),
        "decode_inflight": defaultdict(int),
        "prefill_inflight": defaultdict(int),
        "service_rate": defaultdict(lambda: RateEstimator(window_sec=RATE_WINDOW)),
        "total_requests": 0,
        "start_time": time.time(),
    }
    
    # RR iterators
    app.state.rr_iters = {
        "prefill": itertools.cycle(range(len(app.state.prefill_clients))),
        "decode": itertools.cycle(range(len(app.state.decode_clients))),
    }
    
    # MPC Controller
    app.state.mpc = MPCController(
        horizon=args.mpc_horizon,
        dt=args.mpc_dt,
        lambda_reg=args.mpc_lambda,
    )
    for c in app.state.decode_clients:
        app.state.mpc.weights[c["id"]] = 1.0
    
    # Routing config
    app.state.routing_policy = args.policy
    app.state.mpc_enabled = args.enable_mpc and HAS_CVXPY
    
    # Start MPC background task if enabled
    if app.state.mpc_enabled:
        asyncio.create_task(app.state.mpc.control_loop(app))
        print(f"MPC enabled with policy: {args.policy}")
    else:
        print(f"MPC disabled, using policy: {args.policy}")
    
    yield
    
    # Cleanup
    for c in app.state.prefill_clients + app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


def select_client(app, service: str):
    """Select a client based on routing policy."""
    pool = app.state.decode_clients if service == "decode" else app.state.prefill_clients
    inflight = app.state.metrics[f"{service}_inflight"]
    mpc_weights = app.state.mpc.weights if app.state.mpc_enabled else {}
    policy = app.state.routing_policy
    
    if policy == "random":
        return select_random(pool, inflight, mpc_weights)
    elif policy == "rr":
        if app.state.mpc_enabled:
            return select_weighted_rr(pool, inflight, mpc_weights, app.state.rr_iters[service])
        return select_round_robin(pool, inflight, mpc_weights, app.state.rr_iters[service])
    elif policy == "po2":
        if app.state.mpc_enabled:
            return select_po2_weighted(pool, inflight, mpc_weights)
        return select_po2_vanilla(pool, inflight, mpc_weights)
    else:
        return select_po2_vanilla(pool, inflight, mpc_weights)


async def handle_request(api: str, request: Request):
    """Handle incoming request."""
    app.state.metrics["arrival_rate"].tick(1)
    app.state.metrics["total_requests"] += 1
    
    req_body = await request.json()
    
    # Prefill phase
    prefill_client = select_client(request.app, "prefill")
    app.state.metrics["prefill_inflight"][prefill_client["id"]] += 1
    
    t0 = time.monotonic()
    try:
        resp = await prefill_client["client"].post(
            f"http://{prefill_client['host']}:{prefill_client['port']}/v1{api}",
            json=req_body
        )
        prefill_result = resp.json()
    except Exception as e:
        app.state.metrics["prefill_inflight"][prefill_client["id"]] -= 1
        return JSONResponse({"error": str(e)}, status_code=500)
    
    app.state.metrics["prefill_latency"].update(time.monotonic() - t0)
    app.state.metrics["prefill_inflight"][prefill_client["id"]] -= 1
    
    # Add KV params
    req_body["kv_transfer_params"] = prefill_result.get("kv_transfer_params", {})
    
    # Decode phase
    decode_client = select_client(request.app, "decode")
    app.state.metrics["decode_inflight"][decode_client["id"]] += 1
    
    async def stream_decode():
        start = time.monotonic()
        try:
            async with decode_client["client"].stream(
                "POST",
                f"http://{decode_client['host']}:{decode_client['port']}/v1{api}",
                json=req_body
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        finally:
            elapsed = time.monotonic() - start
            app.state.metrics["decode_latency"].update(elapsed)
            app.state.metrics["decode_inflight"][decode_client["id"]] -= 1
            app.state.metrics["service_rate"][decode_client["id"]].tick(1)
    
    return StreamingResponse(stream_decode(), media_type="application/x-ndjson")


@app.post("/v1/completions")
async def completions(req: Request):
    return await handle_request("/completions", req)


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    return await handle_request("/chat/completions", req)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    m = app.state.metrics
    elapsed = time.time() - m["start_time"]
    
    return {
        "routing_policy": app.state.routing_policy,
        "mpc_enabled": app.state.mpc_enabled,
        "total_requests": m["total_requests"],
        "elapsed_seconds": elapsed,
        "requests_per_second": m["total_requests"] / elapsed if elapsed > 0 else 0,
        "arrival_rate": m["arrival_rate"].get(),
        "prefill_inflight": dict(m["prefill_inflight"]),
        "decode_inflight": dict(m["decode_inflight"]),
        "mpc_weights": dict(app.state.mpc.weights),
        "service_rates": {k: v.get() for k, v in m["service_rate"].items()},
        "latency": {
            "prefill_ms": m["prefill_latency"].get() * 1000,
            "decode_ms": m["decode_latency"].get() * 1000,
        },
        "prefill_servers": len(app.state.prefill_clients),
        "decode_servers": len(app.state.decode_clients),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Orbit Router with MPC")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefiller-hosts", nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", nargs="+", type=int, default=[8100])
    parser.add_argument("--decoder-hosts", nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", nargs="+", type=int, default=[8200])
    
    # Routing policy
    parser.add_argument("--policy", type=str, default="po2",
                        choices=["random", "rr", "po2"],
                        help="Base routing policy")
    parser.add_argument("--enable-mpc", action="store_true", default=False,
                        help="Enable MPC weight adjustment")
    parser.add_argument("--disable-mpc", action="store_false", dest="enable_mpc",
                        help="Disable MPC (use baseline policy)")
    
    # MPC parameters
    parser.add_argument("--mpc-horizon", type=int, default=10,
                        help="MPC prediction horizon")
    parser.add_argument("--mpc-dt", type=float, default=0.1,
                        help="MPC control interval (seconds)")
    parser.add_argument("--mpc-lambda", type=float, default=0.5,
                        help="MPC regularization coefficient")
    
    args = parser.parse_args()
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    
    return args


if __name__ == "__main__":
    global_args = parse_args()
    
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
