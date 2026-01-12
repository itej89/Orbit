import asyncio
import pytest
from collections import defaultdict
import time

from orbit import RateEstimator, WeightedMovingAverager, mpc_control_loop

@pytest.mark.asyncio
async def test_rate_estimator():
    r = RateEstimator(window_sec=0.1)
    for _ in range(10):
        r.tick()
        await asyncio.sleep(0.15)
    assert r.get() > 0, "Rate should be > 0 after ticks"

@pytest.mark.asyncio
async def test_mpc_weights_update():
    class DummyApp:
        def __init__(self):
            self.state = type("", (), {})()
            self.state.decode_inflight = {0: 5, 1: 8}
            self.state.policy = {"node_weights": {0:1.0, 1:1.0}}
            self.state.metrics = {
                "arrival_rate": RateEstimator(window_sec=0.1),
                "service_rate": {0: RateEstimator(window_sec=0.1), 1: RateEstimator(window_sec=0.1)}
            }

    app = DummyApp()
    app.state.metrics["arrival_rate"].tick(5)
    app.state.metrics["service_rate"][0].tick(3)
    app.state.metrics["service_rate"][1].tick(7)
    
    task = asyncio.create_task(mpc_control_loop(app))
    await asyncio.sleep(0.5)
    task.cancel()
    
    # Check weights updated
    assert 0 in app.state.policy["node_weights"]
    assert 1 in app.state.policy["node_weights"]
    assert all(w > 0 for w in app.state.policy["node_weights"].values())
