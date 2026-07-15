#!/usr/bin/env python3
"""
Orbit Router — extended with Kalman-PO2 policy.

New policy: kalman_po2
  - Maintains a per-server Kalman filter on observed request latency
  - PO2 sampling (sample 2 random servers), pick the one with lower
    Kalman-estimated latency rather than weighted inflight count
  - No QP, no arrival rate estimation, no regularization bias
  - Adaptive gain: responds fast when uncertain, smooths when confident
  - Directly measures what matters (latency), not a proxy (queue depth)

Why this should fix MPC's failure:
  - EMA service rate (MPC) is masked when PO2 already balances queues
  - Latency observation is discriminative even when queues are balanced:
    the slow server (max-seqs=8) takes longer per request regardless of
    queue depth, because it has fewer parallel slots
  - Kalman gain naturally decays as estimates converge, giving cold-start
    adaptation without the EMA's fixed alpha
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
# Kalman Filter for per-server latency estimation
# ---------------------------------------------------------------------------
# State model: server latency L_i is a slowly drifting constant
#   L_{k+1} = L_k + w_k,  w_k ~ N(0, Q)   (process noise)
#   z_k     = L_k + v_k,  v_k ~ N(0, R)   (measurement noise)
#
# Tuning rationale:
#   Q = 400 ms^2 → process std ~20ms per step; latency drifts slowly (thermal)
#   R = 3600 ms^2 → measurement std ~60ms; individual requests vary due to batching
#   L0 = 350 ms → conservative initial estimate (slightly above PO2 steady state)
#   P0 = 40000 ms^2 → std ~200ms initial uncertainty; trust data quickly at start
#
# Gain behavior:
#   - Early requests: K ≈ P0/(P0+R) ≈ 0.92 → near-raw observations (fast cold start)
#   - Steady state: K → Q/(Q+R) ≈ 0.10 → smooth like EMA(alpha=0.1) but adaptive

KF_Q  = 400.0    # process noise variance (ms^2)
KF_R  = 3600.0   # measurement noise variance (ms^2)
KF_L0 = 350.0    # initial latency estimate (ms)
KF_P0 = 40000.0  # initial error variance (ms^2)


class KalmanLatency:
    """Single-server Kalman filter on observed request latency."""
    def __init__(self):
        self.L = KF_L0    # latency estimate (ms)
        self.P = KF_P0    # error variance
        self.n = 0        # number of observations

    def update(self, observed_ms: float):
        # Predict step (time update) — process noise accumulates
        P_pred = self.P + KF_Q
        # Kalman gain
        K = P_pred / (P_pred + KF_R)
        # Measurement update
        self.L = self.L + K * (observed_ms - self.L)
        self.P = (1.0 - K) * P_pred
        self.n += 1
        logger.debug("KF server update: obs=%.1fms → L̂=%.1fms P=%.1f K=%.3f n=%d",
                     observed_ms, self.L, self.P, K, self.n)

    @property
    def estimate(self) -> float:
        return self.L

    @property
    def uncertainty(self) -> float:
        """Return std dev of current estimate (ms)."""
        return self.P ** 0.5


# ---------------------------------------------------------------------------
# Estimators (unchanged from original)
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
# MPC control loop (unchanged)
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
    app.state.mode = global_args.mode
    app.state.pool = []
    app.state.prefill_clients = []
    app.state.decode_clients = []

    app.state.metrics = {
        "arrival_rate": RateEstimator(window_sec=RATE_WINDOW),
        "latency": EWMAEstimator(),
        "inflight": defaultdict(int),
        "service_rate": defaultdict(lambda: RateEstimator(window_sec=RATE_WINDOW)),
        "decode_latency": EWMAEstimator(),
        "prefill_latency": EWMAEstimator(),
        "decode_inflight": defaultdict(int),
        "prefill_inflight": defaultdict(int),
        # Kalman filter instances: server_id → KalmanLatency
        "kf_latency": {},
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
            # Initialize Kalman filter for each server
            app.state.metrics["kf_latency"][i] = KalmanLatency()
        logger.info(
            "Single-pool mode: %d servers, policy=%s, MPC=%s",
            len(app.state.pool), global_args.policy, app.state.policy["mpc_enabled"]
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
            app.state.metrics["kf_latency"][i] = KalmanLatency()
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
# Routing policies
# ---------------------------------------------------------------------------

_rr_counter = itertools.count()


def round_robin(pool):
    return pool[next(_rr_counter) % len(pool)]


def least_outstanding(pool, inflight):
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


def kalman_po2(pool, kf_latency):
    """
    Kalman-PO2: Sample 2 servers, pick the one with lower Kalman-estimated latency.

    Unlike MPC-PO2 which uses queue_depth/weight as the routing signal,
    Kalman-PO2 uses estimated request latency — a signal that stays
    discriminative even when queues are balanced, because the slow server
    (fewer max-seqs) takes longer per request regardless of queue depth.

    No QP, no arrival rate estimation. Just: which server do I expect to
    finish my request faster?
    """
    if len(pool) == 1:
        return pool[0]
    a, b = random.sample(pool, 2)
    kf_a = kf_latency.get(a["id"])
    kf_b = kf_latency.get(b["id"])
    L_a = kf_a.estimate if kf_a else KF_L0
    L_b = kf_b.estimate if kf_b else KF_L0
    chosen = a if L_a <= L_b else b
    logger.debug("kalman_po2: server %d(L̂=%.0fms) vs %d(L̂=%.0fms) → %d",
                 a["id"], L_a, b["id"], L_b, chosen["id"])
    return chosen


def prefix_aware(pool, inflight, weights, messages):
    import hashlib
    prefix = ""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            prefix = content[:128]
            break
    if not prefix:
        return weighted_po2(pool, inflight, weights)
    h = int(hashlib.md5(prefix.encode()).hexdigest(), 16)
    preferred = pool[h % len(pool)]
    alt = random.choice([s for s in pool if s["id"] != preferred["id"]] or pool)
    q_pref = inflight[preferred["id"]] / max(weights.get(preferred["id"], 1.0), 1e-6)
    q_alt  = inflight[alt["id"]]      / max(weights.get(alt["id"], 1.0), 1e-6)
    return preferred if q_pref <= q_alt * 2 else alt


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

async def handle_single(api: str, request: Request):
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
    elif policy == "kalman_po2":
        server = kalman_po2(
            app.state.pool,
            app.state.metrics["kf_latency"],
        )
    else:
        # po2 or mpc_po2
        server = weighted_po2(
            app.state.pool,
            app.state.metrics["inflight"],
            app.state.policy["node_weights"],
        )

    sid = server["id"]
    app.state.metrics["inflight"][sid] += 1
    logger.info("→ server %d (%s:%s) [%s]", sid, server["host"], server["port"], policy)

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
            elapsed_ms = (time.monotonic() - start) * 1000
            app.state.metrics["latency"].update(elapsed_ms / 1000)
            app.state.metrics["inflight"][sid] -= 1
            app.state.metrics["service_rate"][sid].tick(1)
            # Update Kalman filter with observed latency
            kf = app.state.metrics["kf_latency"].get(sid)
            if kf is not None:
                kf.update(elapsed_ms)

    return StreamingResponse(gen(), media_type="application/json")


async def handle_pd(api: str, request: Request):
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
            elapsed_ms = (time.monotonic() - start) * 1000
            app.state.metrics["decode_latency"].update(elapsed_ms / 1000)
            app.state.metrics["decode_inflight"][dec["id"]] -= 1
            app.state.metrics["service_rate"][dec["id"]].tick(1)
            kf = app.state.metrics["kf_latency"].get(dec["id"])
            if kf is not None:
                kf.update(elapsed_ms)

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
        servers = []
        for s in a.pool:
            kf = a.metrics["kf_latency"].get(s["id"])
            servers.append({
                "id": s["id"], "host": s["host"], "port": s["port"],
                "inflight": a.metrics["inflight"][s["id"]],
                "weight": a.policy["node_weights"].get(s["id"], 1.0),
                "service_rate": a.metrics["service_rate"][s["id"]].rate,
                "kf_latency_ms": round(kf.estimate, 1) if kf else None,
                "kf_uncertainty_ms": round(kf.uncertainty, 1) if kf else None,
                "kf_n_obs": kf.n if kf else 0,
            })
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
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Orbit Router (with Kalman-PO2)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--enable-mpc", action="store_true")
    parser.add_argument(
        "--policy",
        choices=["rr", "po2", "lor", "prefix", "mpc_po2", "kalman_po2"],
        default="po2",
        help=(
            "rr=round-robin, po2=power-of-2, lor=least-outstanding, "
            "mpc_po2=MPC-augmented PO2, kalman_po2=Kalman-latency PO2"
        ),
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("single")
    sp.add_argument("--servers", nargs="+", required=True, metavar="HOST:PORT")

    pd = sub.add_parser("pd")
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
