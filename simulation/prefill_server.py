import argparse
import asyncio
import random
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/v1/chat/completions")
async def prefill(req: Request):
    # Simulate variable prefill cost
    delay = app.state.delay
    await asyncio.sleep(delay)

    return {
        "kv_transfer_params": {
            "dummy": True
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--delay", type=float, default=0.01)
    args = parser.parse_args()

    app.state.delay = args.delay

    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
