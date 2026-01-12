import asyncio
import httpx
import time
import random
import math

URL = "http://127.0.0.1:8000/v1/chat/completions"
CONCURRENCY = 256
REQUESTS_PER_WORKER = 1024

# --- Traffic parameters ---
MEAN_RPS = 50      # average requests per second per worker
BURSTINESS = 5.0   # controls the variability of arrivals


def poisson_sleep(mean_interval):
    """
    Returns a sleep interval drawn from an exponential distribution to simulate Poisson arrivals.
    """
    return random.expovariate(1.0 / mean_interval)


def diurnal_factor():
    """
    Simulates time-of-day traffic patterns (0..1), just as a simple sine wave.
    """
    t = time.time() % 86400  # seconds in a day
    # peak at midday, low at midnight
    return 0.5 + 0.5 * math.sin(2 * math.pi * t / 86400)


async def send_request(client, wid):
    for i in range(REQUESTS_PER_WORKER):
        # compute interval based on mean RPS, burstiness, and diurnal pattern
        mean_interval = 1.0 / (MEAN_RPS * diurnal_factor())
        sleep_time = poisson_sleep(mean_interval) * random.uniform(0.5, 1.5)  # add some randomness
        await asyncio.sleep(sleep_time)

        t0 = time.monotonic()
        r = await client.post(
            URL,
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            },
        )
        r.raise_for_status()
        latency = time.monotonic() - t0
        # Optionally log latency per worker
        # print(f"[worker {wid}] request {i} latency={latency:.3f}s")


async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            asyncio.create_task(send_request(client, i))
            for i in range(CONCURRENCY)
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
