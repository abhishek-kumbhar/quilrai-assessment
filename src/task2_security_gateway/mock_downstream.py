"""Mock downstream MCP server speaking plain JSON-RPC over HTTP POST.

Run: uv run python -m task2_security_gateway.mock_downstream  (listens on 127.0.0.1:9002)
"""

import logging
import sys

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger("mock_downstream")

TOOLS = [
    {
        "name": "get_customer_record",
        "description": "Look up a customer record.",
        "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate a customer's API key. Admin only.",
        "inputSchema": {"type": "object", "properties": {"customer_id": {"type": "string"}}, "required": ["customer_id"]},
    },
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}

app = FastAPI(title="Mock downstream MCP server")


def reply(request_id, result=None, error=None) -> JSONResponse:
    body = {"jsonrpc": "2.0", "id": request_id}
    body.update({"error": error} if error else {"result": result})
    return JSONResponse(body, headers={"mcp-session-id": "mock-session-1"})


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    message = await request.json()
    if "id" not in message:
        return Response(status_code=202)

    method, request_id = message.get("method"), message["id"]
    logger.info("downstream received method=%s", method)

    if method == "initialize":
        return reply(
            request_id,
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-downstream", "version": "0.1.0"},
            },
        )
    if method == "ping":
        return reply(request_id, {})
    if method == "tools/list":
        return reply(request_id, {"tools": TOOLS})
    if method == "tools/call":
        name = message.get("params", {}).get("name")
        if name not in TOOL_NAMES:
            return reply(request_id, error={"code": -32602, "message": f"Unknown tool: {name}"})
        arguments = message["params"].get("arguments", {})
        return reply(request_id, {"content": [{"type": "text", "text": f"{name} executed with {arguments}"}], "isError": False})
    return reply(request_id, error={"code": -32601, "message": "Method not found"})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=9002)
