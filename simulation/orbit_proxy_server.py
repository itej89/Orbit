#!/usr/bin/env python3
"""
Orbit MPC Proxy Server for Disaggregated vLLM

Drop-in replacement for toy_proxy_server.py with configurable routing policies
and MPC-augmented load balancing.

Routing policies:
  rr     - Round-robin (same as toy_proxy_server.py)
  random - Random server selection
  po2    - Power-of-Two-Choices (pick 2 random, route to less loaded)
  mpc_rr - MPC-weighted round-robin
  mpc_po2- MPC-augmented Power-of-Two-Choices

The proxy follows the same two-phase protocol as toy_proxy_server.py:
  1. Send to prefill with do_remote_decode=True, max_tokens=1
  2. Receive kv_transfer_params
  3. Forward to decode with the kv_transfer_params
"""

import argparse
import asyncio
import itertools
import json
import logging
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

try:
    import cvxpy as cp
    import numpy as np
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orbit_proxy")


@dataclass
class ServerStats:
    """Per-server metrics for routing decisions."""
    host: str
    port: int
    idx: int
    inflight: int = 0
    completed: int = 0
    total_latency: float = 0.0
    capacity_estimate: float = 1.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def avg_latency(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.total_latency / self.completed


class MPCOptimizer:
    """Solves a receding-horizon QP to compute routing weights."""

    def __init__(self, horizon: int = 10, dt: float = 0.1, lambda_reg: float = 0.5):
        self.horizon = horizon
        self.dt = dt
        self.lambda_reg = lambda_reg
        self.weights: Dict[str, float] = {}
        self.last_update = 0.0

    def compute(self, servers: List[ServerStats], arrival_rate: float) -> Dict[str, float]:
        if not HAS_CVXPY or len(servers) < 2:
            return {s.url: 1.0 for s in servers}

        n = len(servers)
        try:
            w = cp.Variable(n, nonneg=True)
            capacities = np.array([max(s.capacity_estimate, 0.1) for s in servers])
            queues = np.array([float(s.inflight) for s in servers])
            current_w = np.array([self.weights.get(s.url, 1.0) for s in servers])

            # Since cp.sum(w) == n, w/n gives fractional allocation (DCP-compliant)
            arrivals = arrival_rate * self.dt * (w / n)
            inv_cap = 1.0 / capacities
            queue_cost = cp.sum_squares(cp.multiply(inv_cap, queues + arrivals))
            reg_cost = self.lambda_reg * cp.sum_squares(w - current_w)

            prob = cp.Problem(
                cp.Minimize(queue_cost + reg_cost),
                [cp.sum(w) == n, w >= 0.1, w <= 3.0],
            )
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if prob.status == cp.OPTIMAL:
                w_opt = w.value / w.value.sum() * n
                self.weights = {s.url: float(w_opt[i]) for i, s in enumerate(servers)}
                self.last_update = time.monotonic()
                return self.weights
        except Exception as e:
            logger.warning(f"MPC solve failed: {e}")

        total_cap = max(sum(s.capacity_estimate for s in servers), 0.01)
        self.weights = {s.url: s.capacity_estimate / total_cap * n for s in servers}
        return self.weights


class OrbitRouter:
    """Routes requests to prefill/decode servers using the configured policy."""

    def __init__(self, policy: str = "rr"):
        self.policy = policy
        self.use_mpc = policy.startswith("mpc_")
        self.base_policy = policy.replace("mpc_", "") if self.use_mpc else policy
        self.mpc_prefill = MPCOptimizer() if self.use_mpc else None
        self.mpc_decode = MPCOptimizer() if self.use_mpc else None
        self.arrival_count = 0
        self.arrival_window_start = time.monotonic()
        self.arrival_rate = 0.0

    def tick_arrival(self):
        self.arrival_count += 1
        elapsed = time.monotonic() - self.arrival_window_start
        if elapsed >= 1.0:
            self.arrival_rate = 0.7 * (self.arrival_count / elapsed) + 0.3 * self.arrival_rate
            self.arrival_count = 0
            self.arrival_window_start = time.monotonic()

    def select(self, servers: List[ServerStats], role: str, rr_iter) -> ServerStats:
        if len(servers) == 1:
            return servers[0]

        if self.base_policy == "random":
            return random.choice(servers)

        if self.base_policy == "rr":
            if self.use_mpc:
                mpc = self.mpc_prefill if role == "prefill" else self.mpc_decode
                weights = mpc.weights
                if weights:
                    weighted = [(s, weights.get(s.url, 1.0)) for s in servers]
                    total = sum(w for _, w in weighted)
                    r = random.uniform(0, total)
                    cumulative = 0.0
                    for s, w in weighted:
                        cumulative += w
                        if r <= cumulative:
                            return s
                    return servers[-1]
            idx = next(rr_iter)
            return servers[idx]

        if self.base_policy == "po2":
            a, b = random.sample(servers, min(2, len(servers)))
            if self.use_mpc:
                mpc = self.mpc_prefill if role == "prefill" else self.mpc_decode
                w = mpc.weights
                score_a = a.inflight / max(w.get(a.url, 1.0), 0.01)
                score_b = b.inflight / max(w.get(b.url, 1.0), 0.01)
            else:
                score_a = a.inflight
                score_b = b.inflight
            return a if score_a <= score_b else b

        return random.choice(servers)

    def update_mpc(self, prefill_servers, decode_servers):
        if self.mpc_prefill:
            self.mpc_prefill.compute(prefill_servers, self.arrival_rate)
        if self.mpc_decode:
            self.mpc_decode.compute(decode_servers, self.arrival_rate)


# Globals
global_args = None
router: Optional[OrbitRouter] = None
prefill_stats: List[ServerStats] = []
decode_stats: List[ServerStats] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, prefill_stats, decode_stats

    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append({
            "client": httpx.AsyncClient(
                timeout=None, base_url=base_url,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            ),
            "host": host, "port": port, "id": i,
        })
        prefill_stats.append(ServerStats(host=host, port=port, idx=i))

    for i, (host, port) in enumerate(global_args.decoder_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append({
            "client": httpx.AsyncClient(
                timeout=None, base_url=base_url,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            ),
            "host": host, "port": port, "id": i,
        })
        decode_stats.append(ServerStats(host=host, port=port, idx=i))

    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    router = OrbitRouter(policy=global_args.policy)

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")
    print(f"Routing policy: {global_args.policy}")

    # MPC weight update task
    async def mpc_update_loop():
        while True:
            if router and router.use_mpc:
                router.update_mpc(prefill_stats, decode_stats)
            await asyncio.sleep(0.5)

    if router.use_mpc:
        asyncio.create_task(mpc_update_loop())

    yield

    for c in app.state.prefill_clients:
        await c["client"].aclose()
    for c in app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(client_info, endpoint, req_data, request_id):
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = await client_info["client"].post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    await response.aread()
    return response


async def stream_service_response(client_info, endpoint, req_data, request_id):
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with client_info["client"].stream("POST", endpoint, json=req_data, headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        router.tick_arrival()

        # Select prefill server
        p_server = router.select(prefill_stats, "prefill", request.app.state.prefill_iterator)
        p_client = request.app.state.prefill_clients[p_server.idx]
        p_server.inflight += 1

        # Send to prefill
        response = await send_request_to_service(p_client, api, req_data, request_id)
        response_json = response.json()
        await response.aclose()

        p_server.inflight = max(0, p_server.inflight - 1)
        prefill_latency = time.monotonic() - start_time
        p_server.completed += 1
        p_server.total_latency += prefill_latency
        if prefill_latency > 0:
            p_server.capacity_estimate = 0.9 * p_server.capacity_estimate + 0.1 / prefill_latency

        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Select decode server
        d_server = router.select(decode_stats, "decode", request.app.state.decode_iterator)
        d_client = request.app.state.decode_clients[d_server.idx]
        d_server.inflight += 1

        decode_start = time.monotonic()

        async def generate_stream():
            try:
                async for chunk in stream_service_response(d_client, api, req_data, request_id):
                    yield chunk
            finally:
                d_server.inflight = max(0, d_server.inflight - 1)
                decode_latency = time.monotonic() - decode_start
                d_server.completed += 1
                d_server.total_latency += decode_latency
                if decode_latency > 0:
                    d_server.capacity_estimate = 0.9 * d_server.capacity_estimate + 0.1 / decode_latency

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        import traceback
        logger.error(f"Error in {api}: {e}")
        traceback.print_exc()
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/health")
@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "policy": global_args.policy if global_args else "unknown",
        "prefill_instances": len(prefill_stats),
        "decode_instances": len(decode_stats),
    }


@app.get("/metrics")
async def metrics():
    return {
        "policy": global_args.policy if global_args else "unknown",
        "arrival_rate": router.arrival_rate if router else 0,
        "prefill": [
            {"url": s.url, "inflight": s.inflight, "completed": s.completed,
             "avg_latency_ms": s.avg_latency * 1000, "capacity": s.capacity_estimate,
             "mpc_weight": router.mpc_prefill.weights.get(s.url, 1.0) if router and router.mpc_prefill else 1.0}
            for s in prefill_stats
        ],
        "decode": [
            {"url": s.url, "inflight": s.inflight, "completed": s.completed,
             "avg_latency_ms": s.avg_latency * 1000, "capacity": s.capacity_estimate,
             "mpc_weight": router.mpc_decode.weights.get(s.url, 1.0) if router and router.mpc_decode else 1.0}
            for s in decode_stats
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Orbit MPC Proxy Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--prefiller-hosts", "--prefiller-host", type=str, nargs="+", required=True)
    parser.add_argument("--prefiller-ports", "--prefiller-port", type=int, nargs="+", required=True)
    parser.add_argument("--decoder-hosts", "--decoder-host", type=str, nargs="+", required=True)
    parser.add_argument("--decoder-ports", "--decoder-port", type=int, nargs="+", required=True)
    parser.add_argument("--policy", type=str, default="rr",
                        choices=["rr", "random", "po2", "mpc_rr", "mpc_po2"],
                        help="Routing policy")
    args = parser.parse_args()
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


if __name__ == "__main__":
    global_args = parse_args()
    print(f"Starting Orbit Proxy on {global_args.host}:{global_args.port}")
    print(f"Policy: {global_args.policy}")
    print(f"Prefill: {global_args.prefiller_instances}")
    print(f"Decode: {global_args.decoder_instances}")
    uvicorn.run(app, host=global_args.host, port=global_args.port)
