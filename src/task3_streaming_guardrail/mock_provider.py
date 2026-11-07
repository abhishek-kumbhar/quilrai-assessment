"""Mock OpenAI-style chat completion provider that leaks PII in small random chunks.

Run: uv run python -m task3_streaming_guardrail.mock_provider  (listens on 127.0.0.1:9003)
"""

import asyncio
import json
import os
import random
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

RESPONSE_TEXT = (
    "Sure! You can reach John at john.doe@example.com or on 555-123-4567. "
    "His SSN is 123-45-6789 and the card on file is 4111 1111 1111 1111. "
    "Order 1234567890123 shipped yesterday. Let me know if you need anything else."
)
CHUNK_DELAY_SECONDS = float(os.environ.get("MOCK_CHUNK_DELAY", "0.02"))

app = FastAPI(title="Mock LLM provider")


def random_chunks(text: str, rng: random.Random) -> list[str]:
    chunks, position = [], 0
    while position < len(text):
        size = rng.randint(1, 6)
        chunks.append(text[position : position + size])
        position += size
    return chunks


def completion_chunk(completion_id: str, model: str, delta: dict, finish_reason=None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-model")
    completion_id = f"chatcmpl-{random.randint(10_000, 99_999)}"

    if not body.get("stream"):
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": RESPONSE_TEXT}, "finish_reason": "stop"}],
            }
        )

    async def events():
        yield completion_chunk(completion_id, model, {"role": "assistant", "content": ""})
        for piece in random_chunks(RESPONSE_TEXT, random.Random()):
            await asyncio.sleep(CHUNK_DELAY_SECONDS)
            yield completion_chunk(completion_id, model, {"content": piece})
        yield completion_chunk(completion_id, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9003)
