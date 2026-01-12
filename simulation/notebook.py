import asyncio
import httpx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

METRICS_URL = "http://10.7.78.208:8000/metrics"  # ORBIT router metrics endpoint
POLL_INTERVAL = 0.5  # seconds

# store history for plotting
history = {
    "time": [],
    "inflight": {},      # node_id -> [values]
    "weights": {},       # node_id -> [values]
    "service_rate": {},  # node_id -> [values]
    "latency_decode": [],
    "latency_prefill": [],
}

start_time = None

async def fetch_metrics(client):
    global start_time
    r = await client.get(METRICS_URL)
    data = r.json()
    print(data)
    if start_time is None:
        start_time = asyncio.get_event_loop().time()
    t = asyncio.get_event_loop().time() - start_time
    history["time"].append(t)

    # decode inflight
    for node, q in data["inflight"].items():
        node = int(node)
        if node not in history["inflight"]:
            history["inflight"][node] = []
        history["inflight"][node].append(q)

    # weights
    for node, w in data["weights"].items():
        node = int(node)
        if node not in history["weights"]:
            history["weights"][node] = []
        history["weights"][node].append(w)

    # service rate
    for node, mu in data["service_rate"].items():
        node = int(node)
        if node not in history["service_rate"]:
            history["service_rate"][node] = []
        history["service_rate"][node].append(mu)

    # latency
    history["latency_decode"].append(data["latency"]["decode"])
    history["latency_prefill"].append(data["latency"]["prefill"])


async def poll_metrics():
    async with httpx.AsyncClient(timeout=1.0) as client:
        while True:
            try:
                await fetch_metrics(client)
            except Exception as e:
                print("Error fetching metrics:", e)
            await asyncio.sleep(POLL_INTERVAL)


def plot_metrics():
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    def update(frame):
        axs[0].cla()
        axs[1].cla()
        axs[2].cla()
        axs[3].cla()

        t = history["time"]

        # Plot inflight
        for node, q in history["inflight"].items():
            axs[0].plot(t, q, label=f"Node {node}")
        axs[0].set_ylabel("Inflight requests")
        axs[0].legend()
        axs[0].grid(True)

        # Plot weights
        for node, w in history["weights"].items():
            axs[1].plot(t, w, label=f"Node {node}")
        axs[1].set_ylabel("MPC Weight")
        axs[1].grid(True)

        # Plot service rate
        for node, mu in history["service_rate"].items():
            axs[2].plot(t, mu, label=f"Node {node}")
        axs[2].set_ylabel("Service rate μ [req/s]")
        axs[2].grid(True)

        # Plot latency
        axs[3].plot(t, history["latency_decode"], label="Decode")
        axs[3].plot(t, history["latency_prefill"], label="Prefill")
        axs[3].set_ylabel("Latency [s]")
        axs[3].set_xlabel("Time [s]")
        axs[3].grid(True)
        axs[3].legend()

        fig.tight_layout()

    ani = FuncAnimation(fig, update, interval=500)
    plt.show()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(poll_metrics())
    plot_metrics()
