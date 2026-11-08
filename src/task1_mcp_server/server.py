"""Customer support MCP server exposed over stdio.

Error handling follows two levels:
- Protocol errors (JSON-RPC `error`): unknown tool or arguments that fail the
  schema -> -32602 Invalid params. The request itself is wrong.
- Tool execution errors (`result.isError = true`): the request was valid but the
  operation failed, e.g. unknown customer or refund above the refundable amount.
"""

import json
import logging
import sys
import uuid
from decimal import Decimal
from typing import Annotated, Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger("task1_mcp_server")

CustomerId = Annotated[str, Field(pattern=r"^CUST-[0-9]{5}$", description="Customer ID, e.g. CUST-00042")]


class GetCustomerRecordInput(BaseModel):
    # strict: no silent coercion ("12.5" -> 12.5); forbid: reject unknown fields
    model_config = ConfigDict(strict=True, extra="forbid")

    customer_id: CustomerId


class TriggerRefundInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    customer_id: CustomerId
    amount: float = Field(gt=0, allow_inf_nan=False, description="Refund amount, must be positive")
    reason: str = Field(min_length=10, description="Reason for the refund, at least 10 characters")


CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Alice Johnson", "email": "alice@example.com", "tier": "gold", "refundable_balance": 500.0},
    "CUST-00002": {"name": "Bob Smith", "email": "bob@example.com", "tier": "standard", "refundable_balance": 75.5},
    "CUST-12345": {"name": "Carol White", "email": "carol@example.com", "tier": "silver", "refundable_balance": 0.0},
}


def get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return tool_error(f"Customer {args.customer_id} not found")
    return tool_success({"customer_id": args.customer_id, **record})


def trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return tool_error(f"Customer {args.customer_id} not found")
    balance = record["refundable_balance"]
    if args.amount > balance:
        return tool_error(f"Refund of {args.amount} exceeds refundable balance of {balance}")

    # Subtract the decimal values as written, so 75.5 - 0.1 leaves 75.4 rather than 75.39999999999999.
    new_balance = float(Decimal(str(balance)) - Decimal(str(args.amount)))
    if new_balance == balance:
        # e.g. 1e-15 off 75.5: below the stored balance's precision, so it can't be recorded.
        return tool_error(f"Refund of {args.amount} is too small to apply to a balance of {balance}")
    record["refundable_balance"] = new_balance
    refund = {
        "refund_id": f"RF-{uuid.uuid4().hex[:10]}",
        "customer_id": args.customer_id,
        "amount": args.amount,
        "reason": args.reason,
        "status": "processed",
    }
    logger.info("refund processed refund_id=%s customer_id=%s", refund["refund_id"], args.customer_id)
    return tool_success(refund)


TOOLS = {
    "get_customer_record": (
        GetCustomerRecordInput,
        get_customer_record,
        "Look up a customer record by customer ID.",
    ),
    "trigger_refund": (
        TriggerRefundInput,
        trigger_refund,
        "Issue a refund to a customer. Amount must be positive and reason at least 10 characters.",
    ),
}


def tool_success(data: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(text=json.dumps(data))],
        structured_content=data,
    )


def tool_error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


def validation_details(error: ValidationError) -> list[dict[str, str]]:
    # Only field path and message; pydantic's raw input echo and docs URLs are noise for clients.
    return [{"field": ".".join(str(part) for part in e["loc"]) or "arguments", "error": e["msg"]} for e in error.errors()]


async def handle_list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(name=name, description=description, input_schema=model.model_json_schema())
            for name, (model, _, description) in TOOLS.items()
        ]
    )


async def handle_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    if params.name not in TOOLS:
        logger.warning("unknown tool requested: %s", params.name)
        raise MCPError(code=types.INVALID_PARAMS, message=f"Unknown tool: {params.name}")

    model, handler, _ = TOOLS[params.name]
    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as exc:
        details = validation_details(exc)
        logger.info("rejected %s call: %s", params.name, details)
        raise MCPError(code=types.INVALID_PARAMS, message=f"Invalid arguments for tool {params.name}", data=details)

    logger.info("calling tool %s", params.name)
    return handler(args)


server = Server(
    "customer-support",
    version="0.1.0",
    on_list_tools=handle_list_tools,
    on_call_tool=handle_call_tool,
)


def request_id_from(error: Exception) -> str | int | None:
    # For a message that parsed as JSON but isn't valid JSON-RPC, pydantic's "missing field"
    # errors carry the whole message as their input, so the id can be recovered from there.
    if not isinstance(error, ValidationError):
        return None
    for detail in error.errors():
        message = detail.get("input")
        if detail["type"] == "missing" and len(detail["loc"]) == 2 and isinstance(message, dict):
            request_id = message.get("id")
            if isinstance(request_id, (str, int)) and not isinstance(request_id, bool):
                return request_id
            return None
    return None


async def forward_parseable_messages(transport_read, server_write, transport_write) -> None:
    # The SDK stdio transport passes unparseable lines on as exceptions, which the
    # server ignores. JSON-RPC 2.0 expects an error reply, so answer those here.
    async with server_write:
        async for item in transport_read:
            if not isinstance(item, Exception):
                await server_write.send(item)
                continue

            is_bad_json = isinstance(item, ValidationError) and item.errors()[0]["type"] == "json_invalid"
            if is_bad_json:
                error = types.ErrorData(code=types.PARSE_ERROR, message="Parse error")
                request_id = None
            else:
                error = types.ErrorData(code=types.INVALID_REQUEST, message="Invalid Request")
                request_id = request_id_from(item)
            logger.warning("rejected malformed message: %s", error.message)
            await transport_write.send(SessionMessage(types.JSONRPCError(jsonrpc="2.0", id=request_id, error=error)))


async def run() -> None:
    async with stdio_server() as (transport_read, transport_write):
        server_write, server_read = anyio.create_memory_object_stream(0)
        async with anyio.create_task_group() as tg:
            tg.start_soon(forward_parseable_messages, transport_read, server_write, transport_write)
            await server.run(server_read, transport_write, server.create_initialization_options())


def main() -> None:
    # stdout carries JSON-RPC only; every log line goes to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("starting customer-support MCP server on stdio")
    anyio.run(run)
