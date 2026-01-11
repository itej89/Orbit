# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

import cvxpy as cp
import numpy as np
import time
import asyncio
import random
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WeightedMovingAverager:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else \
            self.alpha * x + (1 - self.alpha) * self.value

    def get(self):
        return 0.0 if self.value is None else float(self.value)

async def mpc_control_loop(app):
    """
    MPC predicts per-node weights based on
    queue length and latency stabilization.
    """
    H = 5
    target_q = 6

    while True:
        inflight = app.state.metrics["decode_inflight"]
        latency = app.state.metrics["decode_latency"].get()

        weights = {}

        for wid, q in inflight.items():
            w = cp.Variable(H)
            qv = cp.Variable(H + 1)

            cost = 0
            constraints = [qv[0] == q]

            for k in range(H):
                constraints += [
                    qv[k + 1] == qv[k] - w[k],
                    w[k] >= 0.3,
                    w[k] <= 1.5,
                    qv[k + 1] >= 0,
                ]
                cost += (
                    cp.square(qv[k + 1] - target_q)
                    + 0.5 * cp.square(w[k] - 1.0)
                )

            prob = cp.Problem(cp.Minimize(cost), constraints)

            try:
                prob.solve(solver=cp.OSQP, warm_start=True)
                weights[wid] = float(w.value[0])
            except Exception:
                weights[wid] = app.state.policy["node_weights"].get(wid, 1.0)

        app.state.policy["node_weights"].update(weights)

        await asyncio.sleep(0.4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prefill_clients = []
    app.state.decode_clients = []

    app.state.metrics = {
        "arrival_rate": WeightedMovingAverager(),
        "decode_latency": WeightedMovingAverager(),
        "prefill_latency": WeightedMovingAverager(),
        "decode_inflight": defaultdict(int),
        "prefill_inflight": defaultdict(int),
    }

    app.state.policy = {
        "routing_mode": "po2",  # rr | po2
        "node_weights": {},     # worker_id -> weight
    }

       # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=prefiller_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=decoder_base_url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "host": host,
                "port": port,
                "id": i,
            }
        )

    for w in app.state.decode_clients:
        app.state.policy["node_weights"][w["id"]] = 1.0

    app.state.rr_iters = {
         # Initialize round-robin iterators
        "prefill": itertools.cycle(range(len(app.state.prefill_clients))),
        "decode": itertools.cycle(range(len(app.state.decode_clients))),
    }

    asyncio.create_task(mpc_control_loop(app))

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    # For decoder instances
    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def weighted_rr(app, service):
    pool = app.state.decode_clients if service == "decode" \
           else app.state.prefill_clients
    it = app.state.rr_iters[service]

    for _ in range(len(pool)):
        idx = next(it)
        w = pool[idx]
        weight = app.state.policy["node_weights"].get(w["id"], 1.0)
        if random.random() <= min(1.0, weight):
            return w
    return pool[idx]


def weighted_po2(app, service):
    pool = app.state.decode_clients if service == "decode" \
           else app.state.prefill_clients
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
    mode = app.state.policy["routing_mode"]
    if mode == "rr":
        return weighted_rr(app, service)
    return weighted_po2(app, service)

async def handle(api: str, request: Request):
    app = request.app
    app.state.metrics["arrival_rate"].update(1)

    req = await request.json()
    rid = str(uuid.uuid4())

    pre = select_client(app, "prefill")
    app.state.metrics["prefill_inflight"][pre["id"]] += 1

    t0 = time.monotonic()
    resp = await pre["client"].post(
        f"http://{pre['host']}:{pre['port']}/v1{api}", json=req
    )
    app.state.metrics["prefill_latency"].update(time.monotonic() - t0)
    app.state.metrics["prefill_inflight"][pre["id"]] -= 1

    req["kv_transfer_params"] = resp.json().get("kv_transfer_params", {})

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
        app.state.metrics["decode_latency"].update(time.monotonic() - start)
        app.state.metrics["decode_inflight"][dec["id"]] -= 1

    return StreamingResponse(gen(), media_type="application/json")


@app.post("/v1/completions")
async def completions(req: Request):
    return await handle("/completions", req)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    return await handle("/chat/completions", req)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)