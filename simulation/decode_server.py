import argparse
import asyncio
import random
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI()

@app.post("/v1/chat/completions")
async def decode(req: Request):

    async def gen():
        # Simulate token-by-token decoding
        for _ in range(10):
            await asyncio.sleep(app.state.token_delay)
            yield b'{"token":"x"}\n'

    return StreamingResponse(gen(), media_type="application/json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-delay", type=float, default=0.02)
    args = parser.parse_args()

    app.state.token_delay = args.token_delay

    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
