"""LLM gateway endpoint: tenant auth, token rate limiting, then primary/secondary routing."""

import asyncio
import hashlib
import logging
import math
import uuid
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from task4_model_router.rate_limiter import RateLimitExceeded, TokenRateLimiter
from task4_model_router.router import ModelRouter, UpstreamError

logger = logging.getLogger("task4_model_router")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Accepted for OpenAI-client compatibility; the router always sends its own provider model.
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0, le=2)


def estimate_tokens(request: CompletionRequest) -> int:
    # Rough upper-bound reservation: ~4 chars per token plus per-message overhead, plus the
    # full completion budget. Corrected with the provider's reported usage afterwards.
    prompt_chars = sum(len(message.content) for message in request.messages)
    return math.ceil(prompt_chars / 4) + 4 * len(request.messages) + request.max_tokens


def tenant_id(api_key: str) -> str:
    # Store a hash, not the raw key, in the usage table.
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def gateway_error(status: int, error_type: str, message: str, request_id: str, headers=None) -> JSONResponse:
    body = {"error": {"type": error_type, "message": message, "request_id": request_id}}
    return JSONResponse(body, status_code=status, headers={"x-request-id": request_id, **(headers or {})})


def validation_message(error: ValidationError) -> str:
    fields = ", ".join(".".join(str(part) for part in e["loc"]) or "body" for e in error.errors())
    return f"Invalid request fields: {fields}"


def create_app(limiter: TokenRateLimiter, router: ModelRouter, tenant_api_keys: set[str]) -> FastAPI:
    allowed_tenants = {tenant_id(key) for key in tenant_api_keys}
    app = FastAPI(title="LLM Gateway model router")

    async def handle_completion(request: Request, request_id: str) -> JSONResponse:
        scheme, _, api_key = request.headers.get("authorization", "").partition(" ")
        tenant = tenant_id(api_key.strip())
        if scheme.lower() != "bearer" or tenant not in allowed_tenants:
            return gateway_error(401, "authentication_error", "Invalid or missing API key", request_id)

        try:
            completion_request = CompletionRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            return gateway_error(400, "invalid_request", validation_message(exc), request_id)

        estimated = estimate_tokens(completion_request)
        if estimated > limiter.limit_tokens:
            return gateway_error(400, "invalid_request", "Request exceeds the per-minute token budget", request_id)

        try:
            reservation = await asyncio.to_thread(limiter.reserve, tenant, estimated)
        except RateLimitExceeded as exc:
            retry_after = max(1, math.ceil(exc.retry_after_seconds))
            return gateway_error(
                429, "rate_limit_exceeded", "Token rate limit exceeded", request_id, {"retry-after": str(retry_after)}
            )

        try:
            completion, provider = await router.complete(completion_request.model_dump(exclude_none=True))
        except UpstreamError as exc:
            # Tenant isn't charged when no provider produced a completion.
            await asyncio.to_thread(limiter.release, reservation)
            logger.error("request_id=%s upstream failure: %s", request_id, exc)
            if exc.retryable:
                return gateway_error(503, "upstream_unavailable", "All model providers are currently unavailable", request_id)
            return gateway_error(502, "upstream_error", "Model provider rejected the request", request_id)

        usage = completion.get("usage") or {}
        actual = usage.get("total_tokens")
        if isinstance(actual, int) and actual >= 0:
            await asyncio.to_thread(limiter.settle, reservation, actual)

        logger.info("request_id=%s provider=%s estimated=%s actual=%s", request_id, provider, estimated, actual)
        return JSONResponse(completion, headers={"x-request-id": request_id, "x-gateway-provider": provider})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        try:
            return await handle_completion(request, request_id)
        except Exception:
            logger.exception("request_id=%s unhandled gateway error", request_id)
            return gateway_error(500, "internal_error", "Internal gateway error", request_id)

    return app
