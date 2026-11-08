"""LLM gateway endpoint that streams provider output back with PII redacted.

Expects OpenAI-style chat completion streaming: one `data: {json}` line per event and a
final `data: [DONE]`. Only the text content of each choice is forwarded (redacted).
"""

import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from task3_streaming_guardrail.redactor import StreamRedactor, redact_text

logger = logging.getLogger("task3_streaming_guardrail")


def gateway_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": error_type, "message": message}}, status_code=status_code)


# Allowlist of what reaches the client. Other fields a provider may send (logprobs, tool calls,
# refusals, audio) also carry generated text that isn't redacted, so they are dropped.
EVENT_FIELDS = ("id", "object", "created", "model", "system_fingerprint", "usage")


def sse_event(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode()


def checked_choices(body: Any, part: str) -> list[dict]:
    """Return body["choices"] if every choice has a shape we can redact, else raise ValueError.

    `part` is "delta" for stream events and "message" for full completions.
    """
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list):
        raise ValueError("missing choices")
    for choice in choices:
        if not isinstance(choice, dict) or type(choice.get("index", 0)) is not int:
            raise ValueError("bad choice")
        holder = choice.get(part)
        content = holder.get("content") if isinstance(holder, dict) else None
        if (holder is not None and not isinstance(holder, dict)) or not isinstance(content, str | None):
            raise ValueError("bad content")
        if not isinstance(choice.get("finish_reason"), str | None):
            raise ValueError("bad finish_reason")
    return choices


def safe_choice(choice: dict, part: str, content: str | None) -> dict:
    holder = choice.get(part) or {}
    safe = {"role": holder["role"]} if isinstance(holder.get("role"), str) else {}
    if content is not None:
        safe["content"] = content
    return {"index": choice.get("index", 0), part: safe, "finish_reason": choice.get("finish_reason")}


def safe_fields(body: dict) -> dict:
    return {key: body[key] for key in EVENT_FIELDS if key in body}


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
                choices = checked_choices(chunk, "delta")
            except ValueError as exc:
                # Fail closed: an event we can't parse may hold text we'd be unable to redact.
                logger.error("provider sent a malformed stream event: %s", exc)
                for event in leftover_events():
                    yield event
                yield sse_event({"error": {"type": "upstream_error", "message": "LLM provider sent an invalid event"}})
                return

            template = {key: chunk[key] for key in ("id", "object", "created", "model") if key in chunk}
            safe_choices = []
            for choice in choices:
                redactor = redactors[choice.get("index", 0)]
                delta = choice.get("delta") or {}
                safe_text = redactor.feed(delta["content"]) if isinstance(delta.get("content"), str) else ""
                if choice.get("finish_reason"):
                    safe_text += redactor.flush()
                content = safe_text if "content" in delta or safe_text else None
                safe_choices.append(safe_choice(choice, "delta", content))
            yield sse_event({**safe_fields(chunk), "choices": safe_choices})
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
                choices = checked_choices(completion, "message")
            except (ValueError, httpx.HTTPError):
                return gateway_error(502, "upstream_error", "LLM provider returned an invalid response")
            finally:
                await upstream.aclose()
            safe_choices = []
            for choice in choices:
                content = (choice.get("message") or {}).get("content")
                safe_choices.append(safe_choice(choice, "message", None if content is None else redact_text(content)))
            return JSONResponse({**safe_fields(completion), "choices": safe_choices})

        async def redacted_events() -> AsyncIterator[bytes]:
            # Close the provider response however the stream ends (done, error or client disconnect).
            try:
                async for event in redact_sse_stream(upstream.aiter_lines()):
                    yield event
            finally:
                await upstream.aclose()

        return StreamingResponse(
            redacted_events(),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    return app
