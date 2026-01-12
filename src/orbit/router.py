# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import time
import uuid
import asyncio
import random
from collections import defaultdict
from contextlib import asynccontextmanager

import httpx
import cvxpy as cp
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# Time constants
# -----------------------------
MPC_DT = 0.4       
RATE_WINDOW = 0.5  

# -----------------------------
# Utilities
# -----------------------------

class WeightedMovingAverager:
    """EWMA for smoothing scalar metrics."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else \
            self.alpha * x + (1 - self.alpha) * self.value

    def get(self):
        return 0.0 if self.value is None else float(self.value)
    
class RateEstimator:
    """Generic rate estimator (requests/sec) using a sliding window / EWMA style."""
    def __init__(self, window_sec=0.5):
        self.window = window_sec
        self.count = 0
        self.last = time.monotonic()
        self.rate = 0.0

    def tick(self, n=1):
        """Increment counter (called when an event occurs)."""
        now = time.monotonic()
        self.count += n
        if now - self.last >= self.window:
            self.rate = self.count / (now - self.last)
            self.count = 0
            self.last = now

    def get(self):
        return self.rate


# -----------------------------
# MPC Controller
# -----------------------------
async def mpc_control_loop(app):
    """
    MPC predicts per-node weights based on
    fluid backlog, arrival rate, and service rate.
    """
    H = 5
    target_q = 6
    min_w, max_w = 0.3, 1.5
    eps = 1e-6

    while True:
        inflight = app.state.metrics["decode_inflight"]
        arrival_rate = app.state.metrics["arrival_rate"].get()
        service_rate = app.state.metrics["service_rate"]

        current_weights = app.state.policy["node_weights"]
        total_weight = max(sum(current_weights.values()), eps)

        new_weights = {}

        for wid, q0 in inflight.items():
            mu = service_rate.get(wid, 0.5)

            # MPC variables: w = routing weight, q = fluid backlog
            w = cp.Variable(H)
            q = cp.Variable(H + 1)
            constraints = [q[0] == q0]  # initial state
            cost = 0

            for k in range(H):
                p_i = w[k] / (total_weight + eps)
                constraints += [
                    q[k + 1] == q[k] + MPC_DT * (arrival_rate * p_i - mu),
                    q[k + 1] >= 0,
                    w[k] >= min_w,
                    w[k] <= max_w,
                ]
                cost += cp.square(q[k + 1] - target_q) + 0.5 * cp.square(w[k] - 1.0)

            prob = cp.Problem(cp.Minimize(cost), constraints)

            try:
                prob.solve(solver=cp.OSQP, warm_start=True)
                new_weights[wid] = float(w.value[0])
            except Exception:
                new_weights[wid] = current_weights.get(wid, 1.0)

        app.state.policy["node_weights"].update(new_weights)
        await asyncio.sleep(MPC_DT)

# -----------------------------
# FastAPI Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    app.state.metrics = {
        "arrival_rate": RateEstimator(window_sec=RATE_WINDOW),
        "decode_latency": WeightedMovingAverager(),
        "prefill_latency": WeightedMovingAverager(),
        "decode_inflight": defaultdict(int),
        "prefill_inflight": defaultdict(int),
        "service_rate": defaultdict(lambda: RateEstimator(window_sec=RATE_WINDOW)),
    }

    app.state.policy = {
        "routing_mode": "po2",  # rr | po2
        "node_weights": {},
    }

    # Prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        app.state.prefill_clients.append({
            "client": httpx.AsyncClient(timeout=None),
            "host": host,
            "port": port,
            "id": i,
        })

    # Decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        app.state.decode_clients.append({
            "client": httpx.AsyncClient(timeout=None),
            "host": host,
            "port": port,
            "id": i,
        })
        app.state.policy["node_weights"][i] = 1.0

    # Round-robin iterators
    app.state.rr_iters = {
        "prefill": itertools.cycle(range(len(app.state.prefill_clients))),
        "decode": itertools.cycle(range(len(app.state.decode_clients))),
    }

    # Start MPC
    asyncio.create_task(mpc_control_loop(app))

    yield

    for c in app.state.prefill_clients + app.state.decode_clients:
        await c["client"].aclose()

app = FastAPI(lifespan=lifespan)

# -----------------------------
# Routing policies
# -----------------------------
def weighted_rr(app, service):
    pool = app.state.decode_clients if service == "decode" else app.state.prefill_clients
    it = app.state.rr_iters[service]

    for _ in range(len(pool)):
        idx = next(it)
        w = pool[idx]
        weight = app.state.policy["node_weights"].get(w["id"], 1.0)
        if random.random() <= min(weight, 1.0):
            return w
    return pool[idx]

def weighted_po2(app, service):
    pool = app.state.decode_clients if service == "decode" else app.state.prefill_clients
    inflight = app.state.metrics[
        "decode_inflight" if service == "decode" else "prefill_inflight"
    ]
    a, b = random.sample(pool, 2)
    wa = app.state.policy["node_weights"].get(a["id"], 1.0)
    wb = app.state.policy["node_weights"].get(b["id"], 1.0)

    score_a = inflight[a["id"]] / wa
    score_b = inflight[b["id"]] / wb

    return a if score_a <= score_b else b

def select_client(app, service):
    if app.state.policy["routing_mode"] == "rr":
        return weighted_rr(app, service)
    return weighted_po2(app, service)

# -----------------------------
# Request handler
# -----------------------------
async def handle(api: str, request: Request):
    app = request.app
    app.state.metrics["arrival_rate"].tick(1)

    req = await request.json()
    rid = str(uuid.uuid4())

    # Prefill
    pre = select_client(app, "prefill")
    app.state.metrics["prefill_inflight"][pre["id"]] += 1

    t0 = time.monotonic()
    resp = await pre["client"].post(
        f"http://{pre['host']}:{pre['port']}/v1{api}", json=req
    )
    app.state.metrics["prefill_latency"].update(time.monotonic() - t0)
    app.state.metrics["prefill_inflight"][pre["id"]] -= 1

    req["kv_transfer_params"] = resp.json().get("kv_transfer_params", {})

    # Decode
    dec = select_client(app, "decode")
    app.state.metrics["decode_inflight"][dec["id"]] += 1

    async def gen():
        start = time.monotonic()
        async with dec["client"].stream(
            "POST",
            f"http://{dec['host']}:{dec['port']}/v1{api}",
            json=req,
        ) as r:
            async for chunk in r.aiter_bytes():
                yield chunk

        # update metrics
        elapsed = time.monotonic() - start
        app.state.metrics["decode_latency"].update(elapsed)
        app.state.metrics["decode_inflight"][dec["id"]] -= 1
        app.state.metrics["service_rate"][dec["id"]].tick(1)

    return StreamingResponse(gen(), media_type="application/json")

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/v1/completions")
async def completions(req: Request):
    return await handle("/completions", req)

@app.post("/v1/chat/completions")
async def chat(req: Request):
    return await handle("/chat/completions", req)

@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "decode_instances": len(app.state.decode_clients),
        "prefill_instances": len(app.state.prefill_clients),
    }


@app.get("/metrics")
def metrics():
    return {
        "inflight": dict(app.state.metrics["decode_inflight"]),
        "weights": dict(app.state.policy["node_weights"]),
        "service_rate": {
            i: est.rate
            for i, est in app.state.metrics["service_rate"].items()
        },
        "latency": {
            "decode": app.state.metrics["decode_latency"].get(),
            "prefill": app.state.metrics["prefill_latency"].get()
        }
    }

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--prefiller-hosts", nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", nargs="+", type=int, default=[8100])

    parser.add_argument("--decoder-hosts", nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", nargs="+", type=int, default=[8200])

    args = parser.parse_args()

    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args

if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
