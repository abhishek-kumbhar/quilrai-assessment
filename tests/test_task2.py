import json
import time

import httpx
import jwt
import pytest

from task2_security_gateway import mock_downstream
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
        (b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": ["admin_reset_key"]}', 200, -32602),
    ],
)
async def test_malformed_requests(client, downstream, body, status, code):
    response = await client.post("/mcp", content=body, headers={**VIEWER, "content-type": "application/json"})

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert downstream.requests == []


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
