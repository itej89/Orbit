#!/usr/bin/env python3
"""
Standard Orbit Router for Multiple vLLM Backends

Routes requests across multiple vLLM servers using configurable policies.
Supports MPC-augmented routing for heterogeneous server configurations.
"""

import argparse
import itertools
import logging
import time
import random
import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Optional: cvxpy for MPC
try:
    import cvxpy as cp
    import numpy as np
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("WARNING: cvxpy not available, MPC disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class MPCController:
    """MPC-based load balancing weights optimizer."""
    
    def __init__(self, backends: List[str], horizon=10, dt=0.1, lambda_reg=0.5):
        self.backends = backends
        self.horizon = horizon
        self.dt = dt
        self.lambda_reg = lambda_reg
        self.weights = {b: 1.0 for b in backends}
        self.enabled = HAS_CVXPY
        
        # Estimated capacities (updated from metrics)
        self.capacities = {b: 1.0 for b in backends}
        
    def update_capacities(self, queue_depths: Dict[str, int], service_rates: Dict[str, float]):
        """Update capacity estimates based on observed service rates."""
        for b in self.backends:
            if b in service_rates and service_rates[b] > 0:
                self.capacities[b] = service_rates[b]
    
    def compute_weights(self, queue_depths: Dict[str, int], arrival_rate: float) -> Dict[str, float]:
        """Compute optimal routing weights using MPC."""
        if not self.enabled or not HAS_CVXPY:
            return self.weights
        
        n = len(self.backends)
        if n == 0:
            return {}
        
        try:
            import numpy as np
            
            # Decision variables: weights for each backend
            w = cp.Variable(n, nonneg=True)
            
            # Parameters
            caps = np.array([max(self.capacities.get(b, 1.0), 0.1) for b in self.backends])
            queues = np.array([queue_depths.get(b, 0) for b in self.backends])
            current_w = np.array([self.weights[b] for b in self.backends])
            
            # Normalize weights to sum to 1
            w_normalized = w / cp.sum(w)
            
            # Predicted arrivals per backend
            arrivals = arrival_rate * self.dt * w_normalized
            
            # Cost: queue imbalance + regularization for smooth weight changes
            cost = cp.sum_squares(queues + arrivals - caps * self.dt) + \
                   self.lambda_reg * cp.sum_squares(w - current_w)
            
            # Constraints
            constraints = [
                cp.sum(w) == n,  # Sum to n (normalized later)
                w >= 0.1,        # Minimum weight
                w <= 3.0,        # Maximum weight
            ]
            
            prob = cp.Problem(cp.Minimize(cost), constraints)
            prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            
            if prob.status == cp.OPTIMAL:
                w_opt = w.value
                # Normalize
                w_opt = w_opt / w_opt.sum() * n
                self.weights = {b: float(w_opt[i]) for i, b in enumerate(self.backends)}
            
        except Exception as e:
            logger.warning(f"MPC solve failed: {e}")
        
        return self.weights


class StandardRouter:
    """Router for multiple vLLM backends with MPC support."""
    
    def __init__(self, backends: List[str], policy: str = "po2", enable_mpc: bool = False):
        self.backends = backends
        self.policy = policy
        self.enable_mpc = enable_mpc
        
        # Per-backend state
        self.inflight = {b: 0 for b in backends}
        self.completed = {b: 0 for b in backends}
        self.latencies = {b: [] for b in backends}
        
        # Rate tracking
        self.arrival_rate = RateEstimator()
        self.service_rates = {b: RateEstimator() for b in backends}
        
        # MPC controller
        self.mpc = MPCController(backends) if enable_mpc else None
        
        # Round-robin counter
        self.rr_counter = itertools.cycle(range(len(backends)))
        
        # HTTP client
        self.client = None
        
    async def init_client(self):
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def close_client(self):
        if self.client:
            await self.client.aclose()
    
    def select_backend(self) -> str:
        """Select a backend based on the routing policy."""
        if len(self.backends) == 1:
            return self.backends[0]
        
        if self.policy == "random":
            return random.choice(self.backends)
        
        elif self.policy == "rr":
            idx = next(self.rr_counter)
            return self.backends[idx]
        
        elif self.policy == "po2":
            # Power-of-Two-Choices
            a, b = random.sample(self.backends, 2)
            
            if self.enable_mpc and self.mpc:
                # Use MPC weights
                w_a = self.mpc.weights.get(a, 1.0)
                w_b = self.mpc.weights.get(b, 1.0)
                score_a = self.inflight[a] / max(w_a, 0.01)
                score_b = self.inflight[b] / max(w_b, 0.01)
            else:
                score_a = self.inflight[a]
                score_b = self.inflight[b]
            
            return a if score_a <= score_b else b
        
        else:
            return random.choice(self.backends)
    
    async def forward_request(self, request: Request):
        """Forward a request to selected backend."""
        self.arrival_rate.tick()
        backend = self.select_backend()
        self.inflight[backend] += 1
        
        start_time = time.monotonic()
        
        try:
            # Get request body
            body = await request.body()
            
            # Forward to backend
            url = f"{backend}/v1/chat/completions"
            headers = {k: v for k, v in request.headers.items() 
                      if k.lower() not in ['host', 'content-length']}
            
            req_data = json.loads(body)
            
            # Check if streaming
            if req_data.get("stream", False):
                return await self._stream_response(backend, url, headers, body, start_time)
            else:
                return await self._non_stream_response(backend, url, headers, body, start_time)
                
        except Exception as e:
            self.inflight[backend] = max(0, self.inflight[backend] - 1)
            logger.error(f"Request failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    async def _stream_response(self, backend, url, headers, body, start_time):
        """Handle streaming response."""
        async def generate():
            try:
                async with self.client.stream("POST", url, headers=headers, content=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                self._record_completion(backend, start_time)
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    async def _non_stream_response(self, backend, url, headers, body, start_time):
        """Handle non-streaming response."""
        try:
            resp = await self.client.post(url, headers=headers, content=body)
            self._record_completion(backend, start_time)
            return JSONResponse(resp.json(), status_code=resp.status_code)
        except Exception as e:
            self._record_completion(backend, start_time)
            raise
    
    def _record_completion(self, backend, start_time):
        """Record request completion metrics."""
        self.inflight[backend] = max(0, self.inflight[backend] - 1)
        self.completed[backend] += 1
        self.service_rates[backend].tick()
        
        latency = time.monotonic() - start_time
        self.latencies[backend].append(latency)
        if len(self.latencies[backend]) > 1000:
            self.latencies[backend] = self.latencies[backend][-500:]
    
    def get_metrics(self) -> Dict:
        """Get router metrics."""
        return {
            "backends": self.backends,
            "policy": self.policy,
            "mpc_enabled": self.enable_mpc,
            "inflight": self.inflight,
            "completed": self.completed,
            "arrival_rate": self.arrival_rate.get(),
            "service_rates": {b: r.get() for b, r in self.service_rates.items()},
            "mpc_weights": self.mpc.weights if self.mpc else {},
        }
    
    async def update_mpc(self):
        """Update MPC weights periodically."""
        while True:
            if self.mpc and self.enable_mpc:
                queue_depths = self.inflight
                arrival_rate = self.arrival_rate.get()
                service_rates = {b: r.get() for b, r in self.service_rates.items()}
                
                self.mpc.update_capacities(queue_depths, service_rates)
                self.mpc.compute_weights(queue_depths, arrival_rate)
                
                logger.debug(f"MPC weights: {self.mpc.weights}")
            
            await asyncio.sleep(0.5)


# Global router instance
router: Optional[StandardRouter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router
    await router.init_client()
    
    # Start MPC update task if enabled
    if router.enable_mpc:
        asyncio.create_task(router.update_mpc())
    
    yield
    await router.close_client()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    return router.get_metrics()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await router.forward_request(request)


@app.post("/v1/completions")
async def completions(request: Request):
    return await router.forward_request(request)


def main():
    global router
    
    parser = argparse.ArgumentParser(description="Standard Orbit Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", action="append", required=True,
                       help="Backend URL (can be specified multiple times)")
    parser.add_argument("--policy", choices=["random", "rr", "po2"], default="po2")
    parser.add_argument("--enable-mpc", action="store_true", help="Enable MPC routing")
    parser.add_argument("--mpc-url", type=str, help="External MPC controller URL (not used)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting router with backends: {args.backend}")
    logger.info(f"Policy: {args.policy}, MPC: {args.enable_mpc}")
    
    router = StandardRouter(
        backends=args.backend,
        policy=args.policy,
        enable_mpc=args.enable_mpc
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
