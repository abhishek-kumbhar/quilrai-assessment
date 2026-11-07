# Project conventions

- One uv project, `src/` layout, one package per task: `task1_mcp_server`, `task2_security_gateway`,
  `task3_streaming_guardrail`, `task4_model_router`. Each runs with `uv run python -m <package>`.
- Tests: `uv run pytest` (pytest-asyncio in auto mode). One test file per task. No network or paid API keys;
  upstreams are mocked with `httpx.MockTransport` or in-repo mock apps.
- Keep code small and explainable. No shared framework between tasks; duplicate a few lines rather than abstract.
- Logs go to stderr. Task 1 stdout is reserved for JSON-RPC.
- Comments only for non-obvious decisions.

# Key decisions

- Task 1 uses MCP SDK v2 low-level `Server` (not `MCPServer`) so schema failures surface as JSON-RPC -32602
  instead of `isError` tool results. Business failures (unknown customer, refund above balance) are `isError` results.
  Customer IDs are `CUST-` + 5 ASCII digits (`[0-9]`, not `\d`, which matches Unicode digits in pydantic).
- SDK stdio transport drops unparseable lines; `forward_parseable_messages` answers them with -32700/-32600.
- Task 2 tokens are HS256 JWTs (`role` claim, `exp`/`sub` required, algorithms pinned). Gateway forwards the
  re-serialized parsed message (prevents duplicate-key parser differentials), never forwards the caller's
  Authorization header, rejects batches, and matches `admin_` case-insensitively. JSON-RPC errors for
  well-formed requests use HTTP 200; auth failures 401; bad envelopes 400; downstream failures 502/504.
