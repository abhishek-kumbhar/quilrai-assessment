import asyncio
import json
import time

import httpx
import jwt
import pytest
from starlette.requests import ClientDisconnect

from task2_security_gateway import gateway, mock_downstream
from task2_security_gateway.auth import issue_token
from task2_security_gateway.gateway import create_app

SECRET = "test-secret-0123456789abcdef0123456789"
DOWNSTREAM_URL = "http://downstream.test/mcp"


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


ADMIN = bearer(issue_token("alice", "admin", SECRET))
VIEWER = bearer(issue_token("bob", "viewer", SECRET))


def rpc(method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def tool_call(name, request_id=1):
    return rpc("tools/call", {"name": name, "arguments": {"customer_id": "CUST-00001"}}, request_id)


class RecordingDownstream:
    """Mock downstream that records every request it receives."""

    def __init__(self, handler=None):
        self.requests: list[httpx.Request] = []
        self.handler = handler or self.echo

    @staticmethod
    def echo(request):
        message = json.loads(request.content)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": {"echo": message}})

    def __call__(self, request):
        self.requests.append(request)
        return self.handler(request)

    def received_messages(self):
        return [json.loads(r.content) for r in self.requests]


def gateway_client(downstream_transport) -> httpx.AsyncClient:
    upstream_client = httpx.AsyncClient(transport=downstream_transport)
    app = create_app(DOWNSTREAM_URL, SECRET, http_client=upstream_client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


@pytest.fixture
def downstream():
    return RecordingDownstream()


@pytest.fixture
async def client(downstream):
    async with gateway_client(httpx.MockTransport(downstream)) as client:
        yield client


async def test_tools_list_is_forwarded_transparently(client, downstream):
    response = await client.post("/mcp", json=rpc("tools/list", request_id=7), headers=VIEWER)

    assert response.status_code == 200
    assert response.json()["result"]["echo"] == rpc("tools/list", request_id=7)
    assert downstream.received_messages() == [rpc("tools/list", request_id=7)]
    assert str(downstream.requests[0].url) == DOWNSTREAM_URL


async def test_caller_token_is_not_passed_downstream(client, downstream):
    await client.post("/mcp", json=rpc("tools/list"), headers=ADMIN)
    assert "authorization" not in downstream.requests[0].headers


@pytest.mark.parametrize("tool_name", ["admin_reset_key", "ADMIN_reset_key", "Admin_delete_user", " admin_reset_key"])
async def test_viewer_cannot_call_admin_tools(client, downstream, tool_name):
    response = await client.post("/mcp", json=tool_call(tool_name, request_id="req-42"), headers=VIEWER)

    assert response.status_code == 200
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": "req-42",
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    assert downstream.requests == []


async def test_admin_can_call_admin_tools(client, downstream):
    response = await client.post("/mcp", json=tool_call("admin_reset_key"), headers=ADMIN)

    assert response.status_code == 200
    assert "result" in response.json()
    assert downstream.received_messages()[0]["params"]["name"] == "admin_reset_key"


async def test_viewer_can_call_regular_tools(client, downstream):
    for name in ["get_customer_record", "list_admin_users", "reset_admin_key"]:
        response = await client.post("/mcp", json=tool_call(name), headers=VIEWER)
        assert "result" in response.json(), name
    assert len(downstream.requests) == 3


EXPIRED = issue_token("bob", "admin", SECRET, ttl_seconds=-10)
WRONG_SECRET = issue_token("bob", "admin", "some-other-secret-0123456789abcdef")
UNKNOWN_ROLE = jwt.encode({"sub": "bob", "role": "superadmin", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256")
MISSING_ROLE = jwt.encode({"sub": "bob", "exp": int(time.time()) + 60}, SECRET, algorithm="HS256")
UNSIGNED = jwt.encode({"sub": "bob", "role": "admin", "exp": int(time.time()) + 60}, None, algorithm="none")


def signed_role(role):
    return jwt.encode({"sub": "bob", "role": role, "exp": int(time.time()) + 60}, SECRET, algorithm="HS256")


def forged_role_token():
    # Swap a viewer token's payload for an admin one while keeping the original signature.
    header, _, signature = issue_token("bob", "viewer", SECRET).split(".")
    admin_payload = issue_token("bob", "admin", "unrelated-secret-0123456789abcdef").split(".")[1]
    return f"{header}.{admin_payload}.{signature}"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer    "},
        {"Authorization": "Basic YWRtaW46YWRtaW4="},
        {"Authorization": "admin"},
        bearer("not-a-jwt"),
        bearer(EXPIRED),
        bearer(WRONG_SECRET),
        bearer(UNKNOWN_ROLE),
        bearer(MISSING_ROLE),
        bearer(UNSIGNED),
        bearer(forged_role_token()),
        # Correctly signed, but the role claim has the wrong type: still a clean 401, not a 500.
        bearer(signed_role([])),
        bearer(signed_role({})),
        bearer(signed_role(["admin"])),
        bearer(signed_role(1)),
        bearer(signed_role(None)),
    ],
)
async def test_invalid_credentials_are_rejected_before_forwarding(client, downstream, headers):
    response = await client.post("/mcp", json=tool_call("admin_reset_key"), headers=headers)

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert response.json()["error"]["code"] == -32000
    assert downstream.requests == []


async def test_bearer_scheme_is_case_insensitive(client):
    token = ADMIN["Authorization"].split(" ", 1)[1]
    response = await client.post("/mcp", json=rpc("tools/list"), headers={"Authorization": f"bearer {token}"})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "body, status, code",
    [
        (b"{not json", 400, -32700),
        (b"", 400, -32700),
        (b'"just a string"', 400, -32600),
        (b'{"id": 1, "method": "tools/list"}', 400, -32600),
        (b'{"jsonrpc": "2.0", "id": 1, "method": 5}', 400, -32600),
        (b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call"}', 200, -32602),
        (b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": 5}}', 200, -32602),
    ],
)
async def test_malformed_requests(client, downstream, body, status, code):
    response = await client.post("/mcp", content=body, headers={**VIEWER, "content-type": "application/json"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert downstream.requests == []


@pytest.mark.parametrize(
    "message, echoed_id",
    [
        ({"jsonrpc": "2.0", "id": [], "method": "tools/list"}, None),
        ({"jsonrpc": "2.0", "id": True, "method": "tools/list"}, None),
        ({"jsonrpc": "2.0", "id": None, "method": "tools/list"}, None),
        ({"jsonrpc": "2.0", "id": 1.5, "method": "tools/list"}, None),
        ({"jsonrpc": "2.0", "id": {"a": 1}, "method": "tools/list"}, None),
        ({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": "x"}, 3),
        ({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": ["admin_reset_key"]}, 3),
        ({"jsonrpc": "2.0", "method": "notifications/initialized", "params": 1}, None),
        ({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "result": {}}, 3),
        ({"jsonrpc": "2.0", "id": 3, "result": {}, "error": {"code": 1, "message": "m"}}, 3),
        ({"jsonrpc": "2.0", "id": 3}, 3),
        ({"jsonrpc": "2.0", "result": {}}, None),
        ({"jsonrpc": "2.0", "id": "s", "error": "boom"}, "s"),
        ({"jsonrpc": "2.0", "id": "s", "error": {"code": "1", "message": "m"}}, "s"),
        ({"jsonrpc": "2.0", "id": "s", "error": {"code": True, "message": "m"}}, "s"),
        ({"jsonrpc": "2.0", "id": "s", "error": {"code": 1}}, "s"),
    ],
)
async def test_invalid_envelopes_are_rejected_before_forwarding(client, downstream, message, echoed_id):
    response = await client.post("/mcp", json=message, headers=ADMIN)

    assert response.status_code == 400
    assert response.json() == {"jsonrpc": "2.0", "id": echoed_id, "error": {"code": -32600, "message": "Invalid Request"}}
    assert downstream.requests == []


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "2.0", "id": "abc", "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": -1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}},
        {"jsonrpc": "2.0", "id": "srv-1", "result": {}},
        {"jsonrpc": "2.0", "id": 7, "error": {"code": -32603, "message": "client failed", "data": [1]}},
    ],
)
async def test_valid_envelopes_are_forwarded(client, downstream, message):
    downstream.handler = lambda request: httpx.Response(202)
    assert (await client.post("/mcp", json=message, headers=VIEWER)).status_code == 202
    assert downstream.received_messages() == [message]


async def test_oversized_body_is_rejected_without_reading_all_of_it(client, downstream):
    chunks_read = []

    async def body():
        for index in range(4):
            chunks_read.append(index)
            yield b" " * 600_000

    response = await client.post("/mcp", content=body(), headers=VIEWER)

    assert response.status_code == 413
    assert response.json()["error"]["code"] == -32600
    assert chunks_read == [0, 1]  # stopped once 1,200,000 bytes passed the 1,000,000 limit
    assert downstream.requests == []


async def test_body_at_the_limit_is_accepted(client, downstream):
    message = json.dumps(rpc("tools/list")).encode()
    body = message + b" " * (1_000_000 - len(message))
    response = await client.post("/mcp", content=body, headers=VIEWER)
    assert response.status_code == 200


async def test_batch_requests_are_rejected_so_admin_calls_cannot_hide_inside(client, downstream):
    batch = [rpc("tools/list", request_id=1), tool_call("admin_reset_key", request_id=2)]
    response = await client.post("/mcp", json=batch, headers=VIEWER)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600
    assert downstream.requests == []


async def test_duplicate_name_keys_cannot_smuggle_admin_tool(client, downstream):
    smuggled = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_customer_record","name":"admin_reset_key"}}'
    response = await client.post("/mcp", content=smuggled, headers=VIEWER)
    assert response.json()["error"]["code"] == -32001

    # Reverse order: allowed, and the downstream only ever sees the name that was authorized.
    reversed_keys = b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","name":"get_customer_record"}}'
    response = await client.post("/mcp", content=reversed_keys, headers=VIEWER)
    assert "result" in response.json()
    assert b"admin_reset_key" not in downstream.requests[0].content


async def test_notifications_and_client_responses_are_forwarded(downstream):
    downstream.handler = lambda request: httpx.Response(202)
    async with gateway_client(httpx.MockTransport(downstream)) as client:
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        assert (await client.post("/mcp", json=notification, headers=VIEWER)).status_code == 202

        client_response = {"jsonrpc": "2.0", "id": "srv-1", "result": {}}
        assert (await client.post("/mcp", json=client_response, headers=VIEWER)).status_code == 202
    assert len(downstream.requests) == 2


@pytest.mark.parametrize(
    "error, status",
    [
        (httpx.ConnectError("connection refused"), 502),
        (httpx.ReadTimeout("timed out"), 504),
    ],
)
async def test_downstream_failures_return_clean_jsonrpc_errors(error, status):
    def failing(request):
        raise error

    async with gateway_client(httpx.MockTransport(failing)) as client:
        response = await client.post("/mcp", json=tool_call("get_customer_record", request_id=9), headers=VIEWER)

    assert response.status_code == status
    body = response.json()
    assert body["id"] == 9
    assert body["error"]["code"] == -32002
    assert "refused" not in body["error"]["message"]


async def test_downstream_error_responses_pass_through_unchanged(downstream):
    downstream.handler = lambda request: httpx.Response(500, text="downstream exploded")
    async with gateway_client(httpx.MockTransport(downstream)) as client:
        response = await client.post("/mcp", json=rpc("tools/list"), headers=VIEWER)
    assert response.status_code == 500


async def test_upstream_response_is_closed_when_its_stream_fails():
    closed = []

    class FailingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"jsonrpc": "2.0", '
            raise httpx.ReadError("connection reset")

        async def aclose(self):
            closed.append(True)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=FailingStream()))
    async with gateway_client(transport) as client:
        # Headers are already sent, so the failure surfaces as a broken response, not a clean body.
        with pytest.raises(httpx.ReadError):
            await client.post("/mcp", json=rpc("tools/list"), headers=VIEWER)
    assert closed == [True]


async def test_upstream_response_is_closed_after_success(downstream):
    closed = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"jsonrpc": "2.0", "id": 1, "result": {}}'

        async def aclose(self):
            closed.append(True)

    downstream.handler = lambda request: httpx.Response(200, stream=Stream())
    async with gateway_client(httpx.MockTransport(downstream)) as client:
        response = await client.post("/mcp", json=rpc("tools/list"), headers=VIEWER)
    assert response.json() == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert closed == [True]


PARSE_ERROR = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}


@pytest.mark.parametrize(
    "body",
    [
        b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": NaN}}',
        b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": Infinity}}',
        b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": -Infinity}}',
        b'{"jsonrpc": "2.0", "id": NaN, "method": "tools/list"}',
        b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": 1e999}}',
        b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"cursor": -1e999}}',
        b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_customer_record", "arguments": {"x": NaN}}}',
    ],
)
async def test_non_standard_json_numbers_are_parse_errors(client, downstream, body):
    # Python's json module accepts these by default and would re-send them as NaN/Infinity tokens.
    response = await client.post("/mcp", content=body, headers=VIEWER)

    assert response.status_code == 400
    assert response.json() == PARSE_ERROR
    assert downstream.requests == []


async def test_authentication_is_still_checked_before_parsing(client, downstream):
    response = await client.post("/mcp", content=b'{"jsonrpc": "2.0", "id": 1, "params": NaN}')
    assert response.status_code == 401
    assert downstream.requests == []


async def test_finite_numbers_are_forwarded_unchanged(client, downstream):
    raw = b'{"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {"a": 1.5, "b": 1e308, "c": -0.0, "d": 12345678901234567890}}'
    response = await client.post("/mcp", content=raw, headers=VIEWER)

    assert response.status_code == 200
    assert downstream.requests[0].content == (
        b'{"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {"a": 1.5, "b": 1e+308, "c": -0.0, "d": 12345678901234567890}}'
    )


class StalledStream(httpx.AsyncByteStream):
    """Sends one chunk and then never finishes; closing it needs an await, like a real connection."""

    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'{"jsonrpc": "2.0", '
        await asyncio.Event().wait()

    async def aclose(self):
        await asyncio.sleep(0)
        self.closed = True


# How the client goes away: (ASGI spec version the server advertises, message being sent, what happens).
# 2.3 is uvicorn's: a disconnect cancels the response. With 2.4 a failed send raises instead.
LOST_CLIENT = {
    "disconnect while headers are sent": ("2.3", "headers", "disconnect"),
    "disconnect while waiting on upstream": ("2.3", "body", "disconnect_after_send"),
    "disconnect while a chunk is sent": ("2.3", "body", "disconnect"),
    "broken pipe on headers": ("2.4", "headers", "broken_pipe"),
    "broken pipe on a chunk": ("2.4", "body", "broken_pipe"),
    "request task cancelled": ("2.3", "body", "cancel"),
}


async def lose_client(app, headers: dict, body: bytes, spec_version: str, stage: str, action: str):
    """Call the app as an ASGI server would, and lose the client at `stage` in the way `action` says."""
    reached, disconnected = asyncio.Event(), asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        message_stage = "headers" if message["type"] == "http.response.start" else "body"
        if message_stage != stage or not message.get("body", True):
            return
        reached.set()
        if action == "broken_pipe":
            raise BrokenPipeError
        if action.startswith("disconnect"):
            disconnected.set()
        if action == "disconnect":
            await asyncio.Event().wait()  # stuck writing to a client that is gone

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "root_path": "",
        "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
        "client": ("127.0.0.1", 5000),
        "server": ("gateway", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    if action == "cancel":
        await reached.wait()
        task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2)  # a hang fails the test (TimeoutError isn't caught)
    except (ClientDisconnect, asyncio.CancelledError):
        pass


@pytest.mark.parametrize("scenario", LOST_CLIENT)
async def test_upstream_response_is_closed_however_the_client_is_lost(scenario):
    stalled = StalledStream()
    responses = [httpx.Response(200, stream=stalled)]
    upstream = httpx.MockTransport(lambda request: responses.pop() if responses else RecordingDownstream.echo(request))
    app = create_app(DOWNSTREAM_URL, SECRET, http_client=httpx.AsyncClient(transport=upstream))

    body = json.dumps(rpc("tools/list")).encode()
    await lose_client(app, {**VIEWER, "content-type": "application/json"}, body, *LOST_CLIENT[scenario])
    assert stalled.closed

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post("/mcp", json=rpc("tools/list", request_id=2), headers=VIEWER)
    assert response.json()["id"] == 2


@pytest.mark.parametrize("fails", [False, True])
async def test_lifespan_closes_only_the_client_it_created(monkeypatch, fails):
    created = []

    class RecordedClient(httpx.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    injected = httpx.AsyncClient()
    monkeypatch.setattr(gateway.httpx, "AsyncClient", RecordedClient)
    owned_app = create_app(DOWNSTREAM_URL, SECRET)
    injected_app = create_app(DOWNSTREAM_URL, SECRET, http_client=injected)

    for app in (owned_app, injected_app):
        try:
            async with app.router.lifespan_context(app):
                if fails:
                    raise RuntimeError("startup failed after the client was created")
        except RuntimeError:
            pass
    assert [client.is_closed for client in created] == [True]
    assert not injected.is_closed
    await injected.aclose()


async def test_sse_responses_and_session_headers_pass_through(downstream):
    sse_body = b'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'
    downstream.handler = lambda request: httpx.Response(
        200, content=sse_body, headers={"content-type": "text/event-stream", "mcp-session-id": "abc123"}
    )
    async with gateway_client(httpx.MockTransport(downstream)) as client:
        response = await client.post(
            "/mcp",
            json=rpc("tools/list"),
            headers={**VIEWER, "mcp-session-id": "abc123", "accept": "application/json, text/event-stream"},
        )

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["mcp-session-id"] == "abc123"
    assert response.content == sse_body
    assert downstream.requests[0].headers["mcp-session-id"] == "abc123"


async def test_end_to_end_with_mock_downstream_server():
    async with gateway_client(httpx.ASGITransport(app=mock_downstream.app)) as client:
        listed = await client.post("/mcp", json=rpc("tools/list"), headers=VIEWER)
        assert {tool["name"] for tool in listed.json()["result"]["tools"]} == {"get_customer_record", "admin_reset_key"}

        denied = await client.post("/mcp", json=tool_call("admin_reset_key"), headers=VIEWER)
        assert denied.json()["error"]["code"] == -32001

        allowed = await client.post("/mcp", json=tool_call("admin_reset_key"), headers=ADMIN)
        assert "admin_reset_key executed" in allowed.json()["result"]["content"][0]["text"]
