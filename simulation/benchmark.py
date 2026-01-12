import asyncio
import httpx
import time

URL = "http://127.0.0.1:8000/v1/chat/completions"
CONCURRENCY = 256
REQUESTS_PER_WORKER = 1024


async def send_request(client, wid):
    for _ in range(REQUESTS_PER_WORKER):
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
        print(f"[worker {wid}] latency={latency:.3f}s")


async def main():
    async with httpx.AsyncClient(timeout=None) as client:
        tasks = [
            asyncio.create_task(send_request(client, i))
            for i in range(CONCURRENCY)
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
