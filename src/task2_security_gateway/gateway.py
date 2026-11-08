"""MCP security gateway: authenticates the caller and blocks admin_ tools for non-admins."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from task2_security_gateway.auth import AuthError, Caller, caller_from_authorization

logger = logging.getLogger("task2_security_gateway")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL_CALL = -32001
DOWNSTREAM_UNAVAILABLE = -32002

MAX_BODY_BYTES = 1_000_000
# The caller's Authorization header is deliberately not forwarded (no token passthrough).
FORWARDED_REQUEST_HEADERS = ("accept", "mcp-session-id", "mcp-protocol-version", "last-event-id")
FORWARDED_RESPONSE_HEADERS = ("content-type", "mcp-session-id")


def jsonrpc_error(request_id: Any, code: int, message: str, status_code: int = 200, headers=None) -> JSONResponse:
    # JSON-RPC errors for well-formed requests use HTTP 200 so MCP clients match them to
    # the request id instead of treating them as transport failures.
    body = {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
    return JSONResponse(body, status_code=status_code, headers=headers)


def is_valid_id(value: Any) -> bool:
    # MCP ids are strings or integers (never null); `type(...) is int` excludes bool.
    return isinstance(value, str) or type(value) is int


def is_valid_message(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return False
    if "id" in message and not is_valid_id(message["id"]):
        return False
    if "method" in message:
        # Request (with id) or notification (without); MCP params are always an object.
        return (
            isinstance(message["method"], str)
            and isinstance(message.get("params", {}), dict)
            and "result" not in message
            and "error" not in message
        )
    # Clients also POST responses to server-initiated requests: an id and exactly one of result/error.
    if "id" not in message or ("result" in message) == ("error" in message):
        return False
    error = message.get("error", {"code": 0, "message": ""})
    return isinstance(error, dict) and type(error.get("code")) is int and isinstance(error.get("message"), str)


async def read_body(request: Request, limit: int) -> bytes | None:
    """Read the request body, or return None as soon as it exceeds `limit` bytes."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


def is_admin_tool(name: str) -> bool:
    # Case-insensitive so "ADMIN_reset_key" can't slip past a downstream that normalizes names.
    return name.strip().lower().startswith("admin_")


def check_tool_call(message: dict, caller: Caller) -> JSONResponse | None:
    request_id = message.get("id")
    params = message.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("name"), str):
        return jsonrpc_error(request_id, INVALID_PARAMS, "Invalid params: tools/call requires a string 'name'")

    tool_name = params["name"]
    if is_admin_tool(tool_name) and caller.role != "admin":
        logger.warning("denied tool=%s subject=%s role=%s", tool_name, caller.subject, caller.role)
        return jsonrpc_error(request_id, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call")

    logger.info("allowed tool=%s subject=%s role=%s", tool_name, caller.subject, caller.role)
    return None


async def forward(client: httpx.AsyncClient, downstream_url: str, message: dict, request: Request):
    headers = {name: request.headers[name] for name in FORWARDED_REQUEST_HEADERS if name in request.headers}
    headers["content-type"] = "application/json"
    # Re-serialize the parsed message instead of forwarding raw bytes, so the downstream
    # sees exactly what was authorized (e.g. no duplicate "name" keys parsed differently).
    upstream_request = client.build_request("POST", downstream_url, content=json.dumps(message), headers=headers)
    request_id = message.get("id")

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        logger.error("downstream timed out url=%s", downstream_url)
        return jsonrpc_error(request_id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server timed out", status_code=504)
    except httpx.HTTPError as exc:
        logger.error("downstream request failed url=%s error=%r", downstream_url, exc)
        return jsonrpc_error(request_id, DOWNSTREAM_UNAVAILABLE, "Downstream MCP server unavailable", status_code=502)

    async def body() -> AsyncIterator[bytes]:
        # Close the upstream response however streaming ends. A read error after the headers are
        # sent propagates, so the server aborts the connection instead of ending a truncated body cleanly.
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    response_headers = {name: upstream.headers[name] for name in FORWARDED_RESPONSE_HEADERS if name in upstream.headers}
    # Stream through so SSE responses from streamable-HTTP servers pass unchanged.
    return StreamingResponse(body(), status_code=upstream.status_code, headers=response_headers)


def create_app(downstream_url: str, jwt_secret: str, http_client: httpx.AsyncClient | None = None) -> FastAPI:
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="MCP Security Gateway", lifespan=lifespan)

    @app.post("/mcp")
    async def proxy(request: Request):
        try:
            caller = caller_from_authorization(request.headers.get("authorization"), jwt_secret)
        except AuthError as exc:
            return jsonrpc_error(
                None, UNAUTHENTICATED, str(exc), status_code=401, headers={"WWW-Authenticate": 'Bearer error="invalid_token"'}
            )

        body = await read_body(request, MAX_BODY_BYTES)
        if body is None:
            return jsonrpc_error(None, INVALID_REQUEST, "Request body too large", status_code=413)

        try:
            message = json.loads(body)
        except ValueError:
            return jsonrpc_error(None, PARSE_ERROR, "Parse error", status_code=400)

        # Batches were dropped from MCP; rejecting them also means every tools/call is checked.
        if isinstance(message, list):
            return jsonrpc_error(None, INVALID_REQUEST, "Batch requests are not supported", status_code=400)
        if not is_valid_message(message):
            request_id = message.get("id") if isinstance(message, dict) and is_valid_id(message.get("id")) else None
            return jsonrpc_error(request_id, INVALID_REQUEST, "Invalid Request", status_code=400)

        if message.get("method") == "tools/call":
            denial = check_tool_call(message, caller)
            if denial is not None:
                return denial

        return await forward(client, downstream_url, message, request)

    return app
