#!/usr/bin/env python3
"""
Orbit Router — two modes:

  1. Single-pool (--servers):  weighted PO2 + MPC across N standalone vLLM servers.
     Each server handles its own prefill+decode. One request → one server.
     Use this for load-balancing experiments with heterogeneous standalone servers.

  2. PD-disagg (--prefiller-* / --decoder-*):  separate prefill and decode pools
     with P2pNcclConnector request_id injection (legacy mode).
"""

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
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# Optional MPC
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

global_args = None

MPC_DT = 0.1
RATE_WINDOW = 0.3


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

class RateEstimator:
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
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value

    def get(self):
        return 0.0 if self.value is None else float(self.value)


# ---------------------------------------------------------------------------
# MPC control loop (shared by both modes, operates on app.state.pool)
# ---------------------------------------------------------------------------

async def mpc_control_loop(app):
    if not HAS_CVXPY:
        return
    H = 10
    min_w, max_w = 0.1, 3.0
    eps = 1e-6
    weight_alpha = 0.25
    max_delta = 0.1

    while True:
        try:
            inflight = dict(app.state.metrics["inflight"])
            arrival_rate = app.state.metrics["arrival_rate"].get()
            service_rate = app.state.metrics["service_rate"]
            current_weights = dict(app.state.policy["node_weights"])
            total_weight = max(sum(current_weights.values()), eps)

            if inflight:
                avg_q = max(1.0, sum(inflight.values()) / len(inflight))
                q_scaled = {node: q / avg_q for node, q in inflight.items()}
            else:
                q_scaled = {}

            target_q_scaled = 1.0
            new_weights = {}

            for wid, q0 in q_scaled.items():
                mu_est = service_rate.get(wid)
                if mu_est is None:
                    continue
                mu = mu_est.get()
                if mu < 1e-6:
                    continue

                w = cp.Variable(H)
                q = cp.Variable(H + 1)
                constraints = [q[0] == q0]
                cost = 0

                for k in range(H):
                    p_i = w[k] / (total_weight + eps)
                    constraints += [
                        q[k + 1] == q[k] + MPC_DT * (arrival_rate * p_i - mu) / max(avg_q, eps),
                        q[k + 1] >= 0,
                        w[k] >= min_w,
                        w[k] <= max_w,
                    ]
                    cost += cp.square(q[k + 1] - target_q_scaled) + 0.07 * cp.square(w[k] - 1.0)

                prob = cp.Problem(cp.Minimize(cost), constraints)
                try:
                    prob.solve(solver=cp.OSQP, warm_start=True)
                    if w.value is not None:
                        w_new = float(w.value[0])
                        w_old = current_weights.get(wid, 1.0)
                        delta = max(min(w_new - w_old, max_delta), -max_delta)
                        w_smooth = w_old + weight_alpha * delta
                        new_weights[wid] = min(max(w_smooth, min_w), max_w)
                except Exception:
                    new_weights[wid] = current_weights.get(wid, 1.0)

            app.state.policy["node_weights"].update(new_weights)
        except Exception as e:
            logger.warning("MPC error: %s", e)

        await asyncio.sleep(MPC_DT)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mode = global_args.mode  # "single" or "pd"
    app.state.pool = []                # unified server pool (single mode)
    app.state.prefill_clients = []     # PD mode
    app.state.decode_clients = []      # PD mode

    app.state.metrics = {
        "arrival_rate": RateEstimator(window_sec=RATE_WINDOW),
        "latency": EWMAEstimator(),
        "inflight": defaultdict(int),
        "service_rate": defaultdict(lambda: RateEstimator(window_sec=RATE_WINDOW)),
        # legacy PD keys kept for metrics endpoint
        "decode_latency": EWMAEstimator(),
        "prefill_latency": EWMAEstimator(),
        "decode_inflight": defaultdict(int),
        "prefill_inflight": defaultdict(int),
    }
    app.state.policy = {
        "node_weights": {},
        "mpc_enabled": global_args.enable_mpc and HAS_CVXPY,
        "routing_policy": global_args.policy,
    }

    if global_args.mode == "single":
        for i, (host, port) in enumerate(global_args.server_instances):
            app.state.pool.append({
                "client": httpx.AsyncClient(timeout=None),
                "host": host, "port": port, "id": i,
            })
            app.state.policy["node_weights"][i] = 1.0
        logger.info(
            "Single-pool mode: %d servers, MPC=%s",
            len(app.state.pool), app.state.policy["mpc_enabled"]
        )
    else:
        for i, (host, port, kv_port) in enumerate(global_args.prefiller_instances):
            app.state.prefill_clients.append({
                "client": httpx.AsyncClient(timeout=None),
                "host": host, "port": port, "kv_port": kv_port, "id": i,
            })
        for i, (host, port, kv_port) in enumerate(global_args.decoder_instances):
            app.state.decode_clients.append({
                "client": httpx.AsyncClient(timeout=None),
                "host": host, "port": port, "kv_port": kv_port, "id": i,
            })
            app.state.policy["node_weights"][i] = 1.0
        logger.info(
            "PD-disagg mode: %d prefill / %d decode, MPC=%s",
            len(app.state.prefill_clients), len(app.state.decode_clients),
            app.state.policy["mpc_enabled"]
        )

    if app.state.policy["mpc_enabled"]:
        asyncio.create_task(mpc_control_loop(app))
        logger.info("MPC control loop started")

    yield

    for c in app.state.pool + app.state.prefill_clients + app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routing: weighted PO2
# ---------------------------------------------------------------------------

_rr_counter = itertools.count()


def round_robin(pool):
    return pool[next(_rr_counter) % len(pool)]


def least_outstanding(pool, inflight):
    """Least Outstanding Requests: pick server with fewest inflight requests."""
    return min(pool, key=lambda s: inflight[s["id"]])


def weighted_po2(pool, inflight, weights):
    if len(pool) == 1:
        return pool[0]
    a, b = random.sample(pool, 2)
    wa = weights.get(a["id"], 1.0)
    wb = weights.get(b["id"], 1.0)
    score_a = inflight[a["id"]] / max(wa, 1e-6)
    score_b = inflight[b["id"]] / max(wb, 1e-6)
    return a if score_a <= score_b else b


def prefix_aware(pool, inflight, weights, messages):
    """Cache-aware routing: hash the first user message prefix to pick a
    candidate server, then verify it is not overloaded vs. a random alternative.
    Falls back to weighted_po2 if the preferred server is > 2x loaded."""
    import hashlib
    prefix = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            prefix = content[:128]  # first 128 chars as cache key
            break
    if not prefix:
        return weighted_po2(pool, inflight, weights)

    h = int(hashlib.md5(prefix.encode()).hexdigest(), 16)
    preferred = pool[h % len(pool)]
    # fallback to po2 if preferred is heavily overloaded
    alt = random.choice([s for s in pool if s["id"] != preferred["id"]] or pool)
    q_pref = inflight[preferred["id"]] / max(weights.get(preferred["id"], 1.0), 1e-6)
    q_alt  = inflight[alt["id"]]      / max(weights.get(alt["id"], 1.0), 1e-6)
    return preferred if q_pref <= q_alt * 2 else alt


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

async def handle_single(api: str, request: Request):
    """Single-pool mode: pick one server, stream the full response."""
    app = request.app
    app.state.metrics["arrival_rate"].tick(1)

    req = await request.json()
    policy = app.state.policy["routing_policy"]
    if policy == "rr":
        server = round_robin(app.state.pool)
    elif policy == "lor":
        server = least_outstanding(app.state.pool, app.state.metrics["inflight"])
    elif policy == "prefix":
        server = prefix_aware(
            app.state.pool,
            app.state.metrics["inflight"],
            app.state.policy["node_weights"],
            req.get("messages", []),
        )
    else:
        server = weighted_po2(
            app.state.pool,
            app.state.metrics["inflight"],
            app.state.policy["node_weights"],
        )

    sid = server["id"]
    app.state.metrics["inflight"][sid] += 1
    logger.info("→ server %d (%s:%s)", sid, server["host"], server["port"])

    async def gen():
        start = time.monotonic()
        try:
            async with server["client"].stream(
                "POST",
                f"http://{server['host']}:{server['port']}/v1{api}",
                json=req,
            ) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        finally:
            elapsed = time.monotonic() - start
            app.state.metrics["latency"].update(elapsed)
            app.state.metrics["inflight"][sid] -= 1
            app.state.metrics["service_rate"][sid].tick(1)

    return StreamingResponse(gen(), media_type="application/json")


async def handle_pd(api: str, request: Request):
    """PD-disagg mode: prefill first, then streaming decode with request_id injection."""
    app = request.app
    app.state.metrics["arrival_rate"].tick(1)

    req = await request.json()
    base_id = str(uuid.uuid4())

    pre = weighted_po2(
        app.state.prefill_clients,
        app.state.metrics["prefill_inflight"],
        {i: 1.0 for i in range(len(app.state.prefill_clients))},
    )
    dec = weighted_po2(
        app.state.decode_clients,
        app.state.metrics["decode_inflight"],
        app.state.policy["node_weights"],
    )

    decode_zmq_addr = f"{dec['host']}:{dec['kv_port']}"
    prefill_zmq_addr = f"{pre['host']}:{pre['kv_port']}"
    prefill_req = dict(req)
    prefill_req["request_id"] = f"chatcmpl-{base_id}___decode_addr_{decode_zmq_addr}"
    decode_request_id = f"chatcmpl-{base_id}___prefill_addr_{prefill_zmq_addr}___"

    app.state.metrics["prefill_inflight"][pre["id"]] += 1
    t0 = time.monotonic()
    try:
        resp = await pre["client"].post(
            f"http://{pre['host']}:{pre['port']}/v1{api}", json=prefill_req
        )
        app.state.metrics["prefill_latency"].update(time.monotonic() - t0)
    finally:
        app.state.metrics["prefill_inflight"][pre["id"]] -= 1

    kv_params = {}
    try:
        kv_params = resp.json().get("kv_transfer_params", {})
    except Exception:
        pass

    decode_req = dict(req)
    decode_req["request_id"] = decode_request_id
    if kv_params:
        decode_req["kv_transfer_params"] = kv_params

    app.state.metrics["decode_inflight"][dec["id"]] += 1

    async def gen():
        start = time.monotonic()
        try:
            async with dec["client"].stream(
                "POST",
                f"http://{dec['host']}:{dec['port']}/v1{api}",
                json=decode_req,
            ) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        finally:
            app.state.metrics["decode_latency"].update(time.monotonic() - start)
            app.state.metrics["decode_inflight"][dec["id"]] -= 1
            app.state.metrics["service_rate"][dec["id"]].tick(1)

    return StreamingResponse(gen(), media_type="application/json")


async def handle(api: str, request: Request):
    if request.app.state.mode == "single":
        return await handle_single(api, request)
    return await handle_pd(api, request)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def completions(req: Request):
    return await handle("/completions", req)


@app.post("/v1/chat/completions")
async def chat(req: Request):
    return await handle("/chat/completions", req)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    a = app.state
    if a.mode == "single":
        servers = [
            {"id": s["id"], "host": s["host"], "port": s["port"],
             "inflight": a.metrics["inflight"][s["id"]],
             "weight": a.policy["node_weights"].get(s["id"], 1.0),
             "service_rate": a.metrics["service_rate"][s["id"]].rate}
            for s in a.pool
        ]
        return {
            "mode": "single",
            "routing_policy": a.policy["routing_policy"],
            "mpc_enabled": a.policy["mpc_enabled"],
            "arrival_rate": a.metrics["arrival_rate"].get(),
            "latency_ms": a.metrics["latency"].get() * 1000,
            "servers": servers,
            "mpc_weights": dict(a.policy["node_weights"]),
        }
    return {
        "mode": "pd",
        "routing_policy": "mpc_po2" if a.policy["mpc_enabled"] else "po2",
        "mpc_enabled": a.policy["mpc_enabled"],
        "arrival_rate": a.metrics["arrival_rate"].get(),
        "prefill_inflight": dict(a.metrics["prefill_inflight"]),
        "decode_inflight": dict(a.metrics["decode_inflight"]),
        "mpc_weights": dict(a.policy["node_weights"]),
        "service_rates": {i: e.rate for i, e in a.metrics["service_rate"].items()},
        "latency": {
            "prefill_ms": a.metrics["prefill_latency"].get() * 1000,
            "decode_ms": a.metrics["decode_latency"].get() * 1000,
        },
        "prefill_servers": len(a.prefill_clients),
        "decode_servers": len(a.decode_clients),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Orbit Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--enable-mpc", action="store_true")
    parser.add_argument("--policy", choices=["rr", "po2", "lor", "prefix", "mpc_po2"], default="po2",
                        help="Routing policy: rr=round-robin, po2=power-of-2, mpc_po2=MPC-augmented PO2")

    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- single-pool mode ----
    sp = sub.add_parser("single", help="Load-balance across standalone vLLM servers")
    sp.add_argument("--servers", nargs="+", required=True,
                    metavar="HOST:PORT",
                    help="Server list e.g. 127.0.0.1:8000 127.0.0.1:8001")

    # ---- PD disagg mode ----
    pd = sub.add_parser("pd", help="PD-disagg mode with P2pNcclConnector request_id injection")
    pd.add_argument("--prefiller-hosts", nargs="+", default=["127.0.0.1"])
    pd.add_argument("--prefiller-ports", nargs="+", type=int, default=[8100])
    pd.add_argument("--prefiller-kv-ports", nargs="+", type=int, default=[14579])
    pd.add_argument("--decoder-hosts", nargs="+", default=["127.0.0.1"])
    pd.add_argument("--decoder-ports", nargs="+", type=int, default=[8200])
    pd.add_argument("--decoder-kv-ports", nargs="+", type=int, default=[14580])

    args = parser.parse_args()

    if args.mode == "single":
        instances = []
        for s in args.servers:
            host, port = s.rsplit(":", 1)
            instances.append((host, int(port)))
        args.server_instances = instances
    else:
        args.prefiller_instances = list(zip(
            args.prefiller_hosts, args.prefiller_ports, args.prefiller_kv_ports))
        args.decoder_instances = list(zip(
            args.decoder_hosts, args.decoder_ports, args.decoder_kv_ports))

    return args


if __name__ == "__main__":
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
