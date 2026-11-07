"""LLM gateway endpoint that streams provider output back with PII redacted.

Expects OpenAI-style chat completion streaming: one `data: {json}` line per event and a
final `data: [DONE]`.
"""

import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from task3_streaming_guardrail.redactor import StreamRedactor, redact_text

logger = logging.getLogger("task3_streaming_guardrail")


def gateway_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": error_type, "message": message}}, status_code=status_code)


def sse_event(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode()


async def redact_sse_stream(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    # One redactor per choice index so n>1 completions don't bleed into each other.
    redactors: dict[int, StreamRedactor] = defaultdict(StreamRedactor)
    template: dict = {}

    def leftover_events() -> list[bytes]:
        events = []
        for index, redactor in redactors.items():
            remaining = redactor.flush()
            if remaining:
                chunk = {**template, "choices": [{"index": index, "delta": {"content": remaining}, "finish_reason": None}]}
                events.append(sse_event(chunk))
        return events

    try:
        async for line in lines:
            if not line.strip():
                continue  # event separators; we write our own
            if not line.startswith("data:"):
                yield f"{line}\n".encode()  # comments / event names pass through
                continue

            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                for event in leftover_events():
                    yield event
                yield sse_event("[DONE]")
                return

            try:
                chunk = json.loads(payload)
            except ValueError:
                yield sse_event(redact_text(payload))
                continue

            template = {key: chunk[key] for key in ("id", "object", "created", "model") if key in chunk}
            for choice in chunk.get("choices", []):
                redactor = redactors[choice.get("index", 0)]
                delta = choice.get("delta") or {}
                safe_text = redactor.feed(delta["content"]) if isinstance(delta.get("content"), str) else ""
                if choice.get("finish_reason"):
                    safe_text += redactor.flush()
                if "content" in delta or safe_text:
                    delta["content"] = safe_text
                    choice["delta"] = delta
            yield sse_event(chunk)
    except httpx.HTTPError as exc:
        logger.error("provider stream failed: %r", exc)
        # Held text belongs to an interrupted answer; flush it redacted, then report the error.
        for event in leftover_events():
            yield event
        yield sse_event({"error": {"type": "upstream_error", "message": "LLM provider stream interrupted"}})
        return

    # Provider closed without [DONE]: still release what we were holding.
    for event in leftover_events():
        yield event


def create_app(provider_url: str, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="LLM Gateway PII guardrail", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return gateway_error(400, "invalid_request", "Request body must be valid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return gateway_error(400, "invalid_request", "'messages' must be a list")

        stream = body.get("stream") is True
        upstream_request = client.build_request("POST", provider_url, json=body)
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            logger.error("provider request failed: %r", exc)
            return gateway_error(502, "upstream_unavailable", "LLM provider unavailable")

        if upstream.status_code != 200:
            logger.error("provider returned status %s", upstream.status_code)
            await upstream.aclose()
            return gateway_error(502, "upstream_error", "LLM provider returned an error")

        if not stream:
            try:
                completion = json.loads(await upstream.aread())
            except (ValueError, httpx.HTTPError):
                return gateway_error(502, "upstream_error", "LLM provider returned an invalid response")
            finally:
                await upstream.aclose()
            for choice in completion.get("choices", []):
                message = choice.get("message") or {}
                if isinstance(message.get("content"), str):
                    message["content"] = redact_text(message["content"])
            return JSONResponse(completion)

        return StreamingResponse(
            redact_sse_stream(upstream.aiter_lines()),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            background=BackgroundTask(upstream.aclose),
        )

    return app
