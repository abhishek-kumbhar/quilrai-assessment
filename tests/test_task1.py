import copy
import json
import math
import subprocess
import sys

import mcp_types as types
import pytest
from mcp.shared.exceptions import MCPError

from task1_mcp_server import server


@pytest.fixture(autouse=True)
def reset_customers():
    original = copy.deepcopy(server.CUSTOMERS)
    yield
    server.CUSTOMERS.clear()
    server.CUSTOMERS.update(original)


async def call(name, arguments):
    return await server.handle_call_tool(None, types.CallToolRequestParams(name=name, arguments=arguments))


async def call_expecting_protocol_error(name, arguments):
    with pytest.raises(MCPError) as exc_info:
        await call(name, arguments)
    assert exc_info.value.code == types.INVALID_PARAMS
    return exc_info.value


VALID_REFUND = {"customer_id": "CUST-00001", "amount": 25.5, "reason": "Damaged item on arrival"}


async def test_list_tools_exposes_strict_schemas():
    result = await server.handle_list_tools(None, None)
    tools = {tool.name: tool for tool in result.tools}

    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund_schema = tools["trigger_refund"].input_schema
    assert refund_schema["additionalProperties"] is False
    assert refund_schema["required"] == ["customer_id", "amount", "reason"]
    assert refund_schema["properties"]["reason"]["minLength"] == 10


async def test_get_customer_record_success():
    result = await call("get_customer_record", {"customer_id": "CUST-00001"})

    assert result.is_error is False
    assert result.structured_content["name"] == "Alice Johnson"
    assert json.loads(result.content[0].text)["customer_id"] == "CUST-00001"


@pytest.mark.parametrize(
    "customer_id",
    [
        "CUST-1234",
        "CUST-123456",
        "cust-12345",
        "CUST-1234A",
        "CUST12345",
        " CUST-12345",
        "CUST-12345\n",
        "CUST-١٢٣٤٥",  # non-ASCII digits
        "",
        12345,
        None,
    ],
)
async def test_get_customer_record_rejects_bad_customer_id(customer_id):
    error = await call_expecting_protocol_error("get_customer_record", {"customer_id": customer_id})
    assert error.data[0]["field"] == "customer_id"


async def test_missing_and_extra_arguments_are_rejected():
    error = await call_expecting_protocol_error("get_customer_record", {})
    assert error.data == [{"field": "customer_id", "error": "Field required"}]

    error = await call_expecting_protocol_error("get_customer_record", None)
    assert error.data[0]["field"] == "customer_id"

    error = await call_expecting_protocol_error("get_customer_record", {"customer_id": "CUST-00001", "admin": True})
    assert error.data[0]["field"] == "admin"


async def test_unknown_customer_is_tool_error_not_protocol_error():
    result = await call("get_customer_record", {"customer_id": "CUST-99999"})

    assert result.is_error is True
    assert "not found" in result.content[0].text


async def test_unknown_tool_is_protocol_error():
    error = await call_expecting_protocol_error("delete_everything", {})
    assert "Unknown tool" in error.message


async def test_trigger_refund_success_updates_balance():
    result = await call("trigger_refund", VALID_REFUND)

    assert result.is_error is False
    assert result.structured_content["status"] == "processed"
    assert result.structured_content["refund_id"].startswith("RF-")
    assert server.CUSTOMERS["CUST-00001"]["refundable_balance"] == 474.5


async def test_trigger_refund_accepts_integer_amount():
    result = await call("trigger_refund", {**VALID_REFUND, "amount": 10})
    assert result.is_error is False


@pytest.mark.parametrize(
    "amount, balance_after",
    [(0.001, 499.999), (10.005, 489.995), (0.1, 499.9), (1e-9, 499.999999999), (500, 0.0)],
)
async def test_any_positive_float_up_to_the_balance_is_refunded_exactly(amount, balance_after):
    # The assessment only asks for a positive float; no hidden 2-decimal rule, no silent rounding.
    result = await call("trigger_refund", {**VALID_REFUND, "amount": amount})

    assert result.is_error is False
    assert result.structured_content["amount"] == amount
    assert server.CUSTOMERS["CUST-00001"]["refundable_balance"] == balance_after


async def test_reason_has_a_minimum_length_but_no_maximum():
    result = await call("trigger_refund", {**VALID_REFUND, "reason": "x" * 5000})
    assert result.is_error is False
    assert result.structured_content["reason"] == "x" * 5000


@pytest.mark.parametrize("amount", [0, -1, -0.01, -1e-9, "10.00", True, None, math.inf, math.nan, [10]])
async def test_trigger_refund_rejects_invalid_amount(amount):
    error = await call_expecting_protocol_error("trigger_refund", {**VALID_REFUND, "amount": amount})
    assert error.data[0]["field"] == "amount"


@pytest.mark.parametrize("reason", ["too short", "", "   short   ", " " * 20, 1234567890, None])
async def test_trigger_refund_rejects_invalid_reason(reason):
    error = await call_expecting_protocol_error("trigger_refund", {**VALID_REFUND, "reason": reason})
    assert error.data[0]["field"] == "reason"


async def test_trigger_refund_reports_every_invalid_field():
    error = await call_expecting_protocol_error("trigger_refund", {"customer_id": "X", "amount": -5, "reason": "no"})
    assert {item["field"] for item in error.data} == {"customer_id", "amount", "reason"}


async def test_refund_above_balance_is_tool_error():
    result = await call("trigger_refund", {**VALID_REFUND, "customer_id": "CUST-00002", "amount": 100.0})

    assert result.is_error is True
    assert "exceeds refundable balance" in result.content[0].text
    assert server.CUSTOMERS["CUST-00002"]["refundable_balance"] == 75.5


class StdioSession:
    """Talks raw JSON-RPC to the server subprocess, one request at a time like a real client."""

    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-m", "task1_mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def send(self, message):
        line = message if isinstance(message, str) else json.dumps(message)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def receive(self):
        line = self.process.stdout.readline()
        assert line, "server closed stdout unexpectedly"
        return json.loads(line)

    def request(self, request_id, method, params=None):
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.receive()
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == request_id
        return response

    def close(self):
        self.process.stdin.close()
        stdout_rest = self.process.stdout.read()
        stderr = self.process.stderr.read()
        self.process.wait(timeout=10)
        return stdout_rest, stderr


@pytest.fixture
def stdio_session():
    session = StdioSession()
    init = session.request(
        1,
        "initialize",
        {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1"}},
    )
    assert init["result"]["serverInfo"]["name"] == "customer-support"
    session.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield session
    if session.process.poll() is None:
        session.process.kill()
    session.process.wait(timeout=10)
    for pipe in (session.process.stdin, session.process.stdout, session.process.stderr):
        pipe.close()


def test_stdio_wire_protocol_and_error_mapping(stdio_session):
    listed = stdio_session.request(2, "tools/list")
    assert len(listed["result"]["tools"]) == 2

    ok = stdio_session.request(3, "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-00002"}})
    assert ok["result"]["isError"] is False

    tool_error = stdio_session.request(
        4, "tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-99999"}}
    )
    assert tool_error["result"]["isError"] is True
    assert "error" not in tool_error

    invalid = stdio_session.request(
        5, "tools/call", {"name": "trigger_refund", "arguments": {"customer_id": "CUST-00001", "amount": -3, "reason": "x"}}
    )
    assert invalid["error"]["code"] == -32602
    assert "result" not in invalid

    unknown = stdio_session.request(6, "tools/call", {"name": "admin_reset", "arguments": {}})
    assert unknown["error"]["code"] == -32602

    missing_method = stdio_session.request(7, "resources/list")
    assert missing_method["error"]["code"] == -32601


def test_stdout_contains_only_json_rpc_and_logs_go_to_stderr(stdio_session):
    stdio_session.request(2, "tools/list")
    stdio_session.request(3, "tools/call", {"name": "trigger_refund", "arguments": VALID_REFUND})
    stdio_session.request(4, "tools/call", {"name": "trigger_refund", "arguments": {"customer_id": "bad"}})

    stdout_rest, stderr = stdio_session.close()

    # Everything already read was parsed as JSON in request(); nothing else may follow.
    assert stdout_rest == ""
    assert "starting customer-support MCP server" in stderr
    assert "refund processed" in stderr
    assert "rejected trigger_refund call" in stderr


def test_malformed_messages_get_standard_errors(stdio_session):
    stdio_session.send("{this is not json")
    assert stdio_session.receive() == {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}

    # Valid JSON but not a valid JSON-RPC message: the id is echoed when it can be read.
    stdio_session.send('{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": "wrong"}')
    assert stdio_session.receive() == {"jsonrpc": "2.0", "id": 2, "error": {"code": -32600, "message": "Invalid Request"}}

    stdio_session.send('{"jsonrpc": "1.0", "id": "req-x", "method": "tools/list"}')
    assert stdio_session.receive()["id"] == "req-x"

    for unusable_id in ['{"jsonrpc": "2.0", "id": true, "method": "tools/list", "params": "x"}', "[1, 2]"]:
        stdio_session.send(unusable_id)
        response = stdio_session.receive()
        assert response["id"] is None
        assert response["error"]["code"] == -32600

    wrong_param_type = stdio_session.request(3, "tools/call", {"name": 5})
    assert wrong_param_type["error"]["code"] == -32602

    # Server keeps serving after bad input.
    assert stdio_session.request(4, "tools/list")["result"]["tools"]


async def test_official_sdk_client_over_stdio():
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=["-m", "task1_mcp_server"])
    async with Client(params) as client:
        tools = await client.list_tools()
        assert {tool.name for tool in tools.tools} == {"get_customer_record", "trigger_refund"}

        result = await client.call_tool("get_customer_record", {"customer_id": "CUST-12345"})
        assert result.structured_content["name"] == "Carol White"

        with pytest.raises(MCPError) as exc_info:
            await client.call_tool("trigger_refund", {"customer_id": "CUST-12345", "amount": 0, "reason": "x"})
        assert exc_info.value.code == -32602
