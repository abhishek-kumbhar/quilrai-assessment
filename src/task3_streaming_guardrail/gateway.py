"""LLM gateway endpoint that streams provider output back with PII redacted.

Expects OpenAI-style chat completion streaming: one `data: {json}` line per event and a
final `data: [DONE]`. Only the redacted text of each choice and a few string metadata fields
are forwarded.
"""

import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import Receive, Scope, Send

from task3_streaming_guardrail.redactor import StreamRedactor, redact_text

logger = logging.getLogger("task3_streaming_guardrail")


def gateway_error(status_code: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"type": error_type, "message": message}}, status_code=status_code)


# Only what is rebuilt below reaches the client. Other provider fields (logprobs, tool calls,
# refusals, audio, debug data) can carry generated text that isn't redacted, so they are dropped.
# So are numeric fields (`usage`, `created`): a number can't be redacted like text, and a card
# number or SSN written as an integer would otherwise pass straight through.
METADATA_STRINGS = ("id", "object", "model", "system_fingerprint")
MAX_CHOICES = 128  # OpenAI's limit on `n`; also keeps `index` too small to carry a number like that
FINISH_REASONS = ("stop", "length", "content_filter", "tool_calls", "function_call")
CLOSE_TIMEOUT_SECONDS = 5


def parse_json(data: str | bytes) -> Any:
    try:
        return json.loads(data)
    except RecursionError:  # nested deeper than the parser can recurse
        raise ValueError("JSON is nested too deeply") from None


def sse_event(payload: dict | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {data}\n\n".encode()


def stream_error(message: str) -> bytes:
    return sse_event({"error": {"type": "upstream_error", "message": message}})


def is_count(value: Any) -> bool:
    return type(value) is int and value >= 0  # `type(...) is int` rejects bool


class MalformedResponse(ValueError):
    """Provider output that doesn't have the shape the redaction relies on."""


def checked_choices(body: Any, part: str) -> list[dict]:
    """Return body["choices"] if every choice has a shape we can redact, else raise MalformedResponse.

    `part` is "delta" for stream events and "message" for full completions.
    """
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list):
        raise MalformedResponse("missing choices")
    for choice in choices:
        index = choice.get("index", 0) if isinstance(choice, dict) else None
        if not is_count(index) or index >= MAX_CHOICES:
            raise MalformedResponse("bad choice")
        holder = choice.get(part)
        content = holder.get("content") if isinstance(holder, dict) else None
        if (holder is not None and not isinstance(holder, dict)) or not isinstance(content, str | None):
            raise MalformedResponse("bad content")
        if choice.get("finish_reason") not in (*FINISH_REASONS, None):  # a tuple: values may be unhashable
            raise MalformedResponse("bad finish_reason")
    return choices


def safe_choice(choice: dict, part: str, content: str | None) -> dict:
    holder = choice.get(part) or {}
    safe = {"role": "assistant"} if holder.get("role") == "assistant" else {}
    if content is not None:
        safe["content"] = content
    return {"index": choice.get("index", 0), part: safe, "finish_reason": choice.get("finish_reason")}


def safe_fields(body: dict) -> dict:
    """Top-level string metadata, redacted: a provider could echo generated text into any of it."""
    return {key: redact_text(body[key]) for key in METADATA_STRINGS if isinstance(body.get(key), str)}


async def close_upstream(upstream: httpx.Response) -> None:
    # Shielded: a client disconnect arrives as cancellation, and an unshielded close would be
    # cancelled at its first await. Bounded so a stuck close can't hang the task.
    with anyio.move_on_after(CLOSE_TIMEOUT_SECONDS, shield=True):
        await upstream.aclose()


class UpstreamStreamingResponse(StreamingResponse):
    """Streams to the client and closes the provider response however sending ends.

    Closing here rather than in the body generator also covers a client lost before the body
    starts (while headers are sent) and a send that raises (ASGI 2.4 `BrokenPipeError`).
    """

    def __init__(self, upstream: httpx.Response, content: AsyncIterator[bytes], **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self.upstream = upstream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await close_upstream(self.upstream)


async def redact_sse_stream(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    # One redactor per choice index so n>1 completions don't bleed into each other.
    redactors: dict[int, StreamRedactor] = defaultdict(StreamRedactor)
    finished: set[int] = set()
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
            # Blank lines are event separators (we write our own). Comments and id:/event:/retry:
            # fields aren't part of the OpenAI format and could carry anything, so they are dropped.
            if not line.startswith("data:"):
                continue

            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                for event in leftover_events():
                    yield event
                yield sse_event("[DONE]")
                return

            try:
                chunk = parse_json(payload)
                choices = checked_choices(chunk, "delta")
            except ValueError as exc:
                # Fail closed: an event we can't parse may hold text we'd be unable to redact. The held
                # tail is dropped too, since it may be the start of a value that will now never complete.
                logger.error("provider sent a malformed stream event: %s", exc)
                yield stream_error("LLM provider sent an invalid event")
                return

            template = safe_fields(chunk)
            safe_choices = []
            for choice in choices:
                index = choice.get("index", 0)
                redactor = redactors[index]
                delta = choice.get("delta") or {}
                safe_text = redactor.feed(delta["content"]) if isinstance(delta.get("content"), str) else ""
                if choice.get("finish_reason"):
                    safe_text += redactor.flush()
                    finished.add(index)
                content = safe_text if "content" in delta or safe_text else None
                safe_choices.append(safe_choice(choice, "delta", content))
            yield sse_event({**template, "choices": safe_choices})
    except httpx.HTTPError as exc:
        logger.error("provider stream failed: %r", exc)
        # As above: the held tail of an interrupted answer is dropped, not flushed.
        yield stream_error("LLM provider stream interrupted")
        return

    # The body ended without [DONE]. That is only a complete answer if every choice got a finish
    # reason (which already flushed it); otherwise the held tail may be half a value, so drop it.
    if not redactors or any(index not in finished or redactor.pending for index, redactor in redactors.items()):
        logger.error("provider stream ended before the answer was complete")
        yield stream_error("LLM provider stream interrupted")


def create_app(provider_url: str, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if http_client is None:  # an injected client belongs to the caller
                await client.aclose()

    app = FastAPI(title="LLM Gateway PII guardrail", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = parse_json(await request.body())
        except ValueError:
            return gateway_error(400, "invalid_request", "Request body must be valid JSON")
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            return gateway_error(400, "invalid_request", "'messages' must be a list")

        stream = body.get("stream") is True
        try:
            upstream_request = client.build_request("POST", provider_url, json=body)
        # NaN/Infinity (or 1e999): Python's parser accepts them, httpx won't send them. Nesting that
        # parsed can still be too deep for httpx to re-encode (seen on Python 3.12).
        except (ValueError, RecursionError):
            return gateway_error(400, "invalid_request", "Request body must be valid JSON")
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            logger.error("provider request failed: %r", exc)
            return gateway_error(502, "upstream_unavailable", "LLM provider unavailable")

        if upstream.status_code != 200:
            logger.error("provider returned status %s", upstream.status_code)
            await close_upstream(upstream)
            return gateway_error(502, "upstream_error", "LLM provider returned an error")

        if not stream:
            try:
                completion = parse_json(await upstream.aread())
                choices = checked_choices(completion, "message")
            except (ValueError, httpx.HTTPError):
                return gateway_error(502, "upstream_error", "LLM provider returned an invalid response")
            finally:
                await close_upstream(upstream)
            safe_choices = []
            for choice in choices:
                content = (choice.get("message") or {}).get("content")
                safe_choices.append(safe_choice(choice, "message", None if content is None else redact_text(content)))
            return JSONResponse({**safe_fields(completion), "choices": safe_choices})

        return UpstreamStreamingResponse(
            upstream,
            redact_sse_stream(upstream.aiter_lines()),
            media_type="text/event-stream",
            headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
        )

    return app
