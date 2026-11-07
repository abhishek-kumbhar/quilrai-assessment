# Project conventions

- Python >=3.12 (no .python-version pinned). One uv project, `src/` layout, one package per task: `task1_mcp_server`, `task2_security_gateway`,
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
- Task 3 `StreamRedactor` holds back only the trailing run of chars that could still extend a match
  (email chars, or digits/space/dash), capped at 256, and never cuts through a match. Invariant tested by fuzz:
  streaming output == `redact_text(full_text)` for any chunking. Digits are matched as whole runs (13-48 digits);
  `redact_digit_run` redacts SSNs and Luhn-valid 13-19 digit group sequences inside them. Every pattern has a max length.
- Task 4 limiter is a sliding log in SQLite (`token_usage` rows), check+insert inside `BEGIN IMMEDIATE`,
  connection per operation (run via `asyncio.to_thread`). Reserve estimate (chars/4 + 4/message + max_tokens),
  settle to provider `usage.total_tokens`, release if every provider fails. Tenant stored as sha256 prefix of key.
  Router wraps the whole primary call in `asyncio.timeout(3.0)`; falls back on 429/timeout/5xx/network/bad JSON,
  not on other 4xx. Clients only ever see `{"error": {type, message, request_id}}`.
