
import asyncio
import httpx
import time
import random
import math
import csv
import signal
import sys

URL = "http://127.0.0.1:8000/v1/chat/completions"
METRICS_URL = "http://127.0.0.1:8000/metrics"

CONCURRENCY = 256
REQUESTS_PER_WORKER = 1024
MEAN_RPS = 50
METRICS_INTERVAL = 1.0  # seconds

stop_event = asyncio.Event()

request_records = asyncio.Queue()
metrics_records = asyncio.Queue()


def poisson_sleep(mean_interval):
    return random.expovariate(1.0 / mean_interval)


def diurnal_factor():
    t = time.time() % 86400
    return 0.5 + 0.5 * math.sin(2 * math.pi * t / 86400)


async def send_requests(client, wid):
    for i in range(REQUESTS_PER_WORKER):
        mean_interval = 1.0 / (MEAN_RPS * diurnal_factor())
        await asyncio.sleep(poisson_sleep(mean_interval))

        t0 = time.monotonic()
        try:
            resp = await client.post(
                URL,
                json={"messages": [{"role": "user", "content": "hello"}], "max_tokens": 16},
            )
            status = resp.status_code
            content_len = len(resp.content)
        except Exception as e:
            # Record transport error as a synthetic row
            status = -1
            content_len = 0

        latency = time.monotonic() - t0

        await request_records.put({
            "timestamp": time.time(),
            "worker_id": wid,
            "latency": latency,
            "status": status,
            "bytes": content_len,
        })


async def poll_metrics(client):
    while not stop_event.is_set():
        t0 = time.time()
        try:
            r = await client.get(METRICS_URL)
            r.raise_for_status()
            data = r.json()

            await metrics_records.put({
                "timestamp": t0,
                "decode_inflight": data.get("inflight"),
                "node_weights": data.get("weights"),
                "service_rate": data.get("service_rate"),
                "decode_latency": (data.get("latency") or {}).get("decode"),
                "prefill_latency": (data.get("latency") or {}).get("prefill"),
            })
        except Exception as e:
            await metrics_records.put({
                "timestamp": t0,
                "decode_inflight": None,
                "node_weights": None,
                "service_rate": None,
                "decode_latency": None,
                "prefill_latency": None,
                "error": str(e),
            })

        await asyncio.sleep(METRICS_INTERVAL)


async def request_writer():
    """
    Stream rows to requests.csv as they arrive. Flush after each write
    to keep the file current on disk.
    """
    fieldnames = ["timestamp", "worker_id", "latency", "status", "bytes"]
    # Open in write mode initially to emit header, then append for subsequent runs if needed
    with open("requests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        # Consume until stop_event is set AND queue is drained
        while True:
            try:
                row = await asyncio.wait_for(request_records.get(), timeout=0.25)
                # Only keep known columns (ignore extras)
                writer.writerow({k: row.get(k) for k in fieldnames})
                f.flush()
                request_records.task_done()
            except asyncio.TimeoutError:
                if stop_event.is_set() and request_records.empty():
                    break


async def metrics_writer():
    """
    Stream rows to metrics.csv as they arrive. Flush after each write.
    """
    fieldnames = [
        "timestamp",
        "decode_inflight",
        "node_weights",
        "service_rate",
        "decode_latency",
        "prefill_latency",
        "error",  # present only when poll_metrics catches an exception
    ]
    with open("metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        while True:
            try:
                row = await asyncio.wait_for(metrics_records.get(), timeout=0.25)
                writer.writerow({k: row.get(k) for k in fieldnames})
                f.flush()
                metrics_records.task_done()
            except asyncio.TimeoutError:
                if stop_event.is_set() and metrics_records.empty():
                    break


def _install_signal_handlers(loop):
    # Graceful shutdown on Ctrl+C / SIGTERM
    def _handle_stop(signame):
        # Set stop flag; poller and writers will drain and finish
        stop_event.set()

    for name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, name), lambda n=name: _handle_stop(n))
        except (NotImplementedError, AttributeError):
            # Not available on all platforms (e.g., Windows event loop variants)
            pass


async def main():
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    async with httpx.AsyncClient(timeout=None) as client:
        # Start streaming writers first so they can consume immediately
        req_writer_task = asyncio.create_task(request_writer())
        met_writer_task = asyncio.create_task(metrics_writer())

        # Start poller
        poller = asyncio.create_task(poll_metrics(client))

        # Start workers
        workers = [
            asyncio.create_task(send_requests(client, i))
            for i in range(CONCURRENCY)
        ]

        # Wait for all workers to finish
        await asyncio.gather(*workers)

        # Signal stop; poller will exit after next sleep tick
        stop_event.set()
        await poller

        # Allow queues to drain then finish writers
        await asyncio.gather(
            request_records.join(),  # ensures all queued items were processed
            metrics_records.join(),
        )
        await asyncio.gather(req_writer_task, met_writer_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Fallback for platforms without proper signal handling
        stop_event.set()
        # Let asyncio.run cleanup
