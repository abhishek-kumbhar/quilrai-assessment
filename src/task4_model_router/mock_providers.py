"""Mock chat completion providers with configurable failure modes.

Run: uv run python -m task4_model_router.mock_providers --name primary --port 9004 --mode slow
Modes: ok, rate_limited (429), slow (sleeps --delay seconds), error (500 with internal details).
"""

import argparse
import asyncio
import math

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def create_provider_app(name: str, mode: str = "ok", delay_seconds: float = 5.0) -> FastAPI:
    app = FastAPI(title=f"Mock provider {name}")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if mode == "rate_limited":
            return JSONResponse({"error": {"message": "Rate limit reached for org-123"}}, status_code=429)
        if mode == "error":
            return JSONResponse({"detail": "Traceback (most recent call last): db at 10.0.3.7 failed"}, status_code=500)
        if mode == "slow":
            await asyncio.sleep(delay_seconds)

        text = f"Hello from {name}."
        prompt_tokens = math.ceil(sum(len(m["content"]) for m in body["messages"]) / 4)
        completion_tokens = math.ceil(len(text) / 4)
        return {
            "id": f"cmpl-{name}",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="primary")
    parser.add_argument("--port", type=int, default=9004)
    parser.add_argument("--mode", choices=["ok", "rate_limited", "slow", "error"], default="ok")
    parser.add_argument("--delay", type=float, default=5.0)
    args = parser.parse_args()
    uvicorn.run(create_provider_app(args.name, args.mode, args.delay), host="127.0.0.1", port=args.port)
