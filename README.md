# QuillrAI Solution Architect assessment

Four small, independent components, one per task:

| Task | Package | What it is |
|------|---------|------------|
| 1 | `task1_mcp_server` | MCP server over stdio with `get_customer_record` and `trigger_refund` |
| 2 | `task2_security_gateway` | HTTP JSON-RPC proxy in front of an MCP server, with JWT roles and `admin_*` tool checks |
| 3 | `task3_streaming_guardrail` | LLM gateway endpoint that streams provider output with emails, SSNs and card numbers redacted |
| 4 | `task4_model_router` | LLM gateway with a per-tenant token rate limit (SQLite) and primary→secondary fallback |

Every upstream (downstream MCP server, LLM providers) is a mock included in the repo, so nothing needs a real API key.

## Stack

Python 3.12+ (developed on 3.14), [uv](https://docs.astral.sh/uv/), official MCP Python SDK (`mcp` 2.x),
FastAPI + uvicorn, Pydantic v2, HTTPX, PyJWT, SQLite (stdlib `sqlite3`), pytest + pytest-asyncio.

## Setup

```bash
uv sync
uv run pytest
```

## Running each task

### Task 1: MCP server (stdio)

```bash
uv run python -m task1_mcp_server
```

It speaks JSON-RPC on stdin/stdout and logs to stderr. A quick manual session (the `sleep` keeps stdin open
long enough for the responses to be written):

```bash
(printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"demo","version":"1"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_customer_record","arguments":{"customer_id":"CUST-00001"}}}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"trigger_refund","arguments":{"customer_id":"CUST-1","amount":-5,"reason":"short"}}}'
 sleep 1) | uv run python -m task1_mcp_server 2>/dev/null
```

Request 3 comes back as `{"error": {"code": -32602, "message": "Invalid arguments for tool trigger_refund", "data": [...]}}`
with one entry per invalid field.

To use it from an MCP client such as Claude Desktop:

```json
{"mcpServers": {"customer-support": {"command": "uv", "args": ["--directory", "/path/to/repo", "run", "python", "-m", "task1_mcp_server"]}}}
```

Mock customers: `CUST-00001`, `CUST-00002`, `CUST-12345`.

### Task 2: MCP security gateway

```bash
# terminal 1: mock downstream MCP server on :9002
uv run python -m task2_security_gateway.mock_downstream
# terminal 2: gateway on :8002 (set GATEWAY_JWT_SECRET; a dev secret is used if unset)
uv run python -m task2_security_gateway
```

```bash
VIEWER=$(uv run python -m task2_security_gateway.auth viewer --subject bob)
ADMIN=$(uv run python -m task2_security_gateway.auth admin --subject alice)

# forwarded
curl -s localhost:8002/mcp -H "Authorization: Bearer $VIEWER" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

# blocked at the gateway: {"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unauthorized Tool Call"}}
curl -s localhost:8002/mcp -H "Authorization: Bearer $VIEWER" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"customer_id":"CUST-00001"}}}'

# allowed
curl -s localhost:8002/mcp -H "Authorization: Bearer $ADMIN" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"customer_id":"CUST-00001"}}}'
```

Settings: `DOWNSTREAM_URL` (default `http://127.0.0.1:9002/mcp`), `GATEWAY_PORT` (8002), `GATEWAY_JWT_SECRET`.
`DOWNSTREAM_URL` can also point at a real MCP streamable-HTTP server; SSE responses and `mcp-session-id`
headers are passed through.

### Task 3: Streaming PII guardrail

```bash
# terminal 1: mock OpenAI-style provider on :9003 (streams text containing PII in 1-6 char chunks)
uv run python -m task3_streaming_guardrail.mock_provider
# terminal 2: gateway on :8003
uv run python -m task3_streaming_guardrail
```

```bash
curl -N localhost:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mock","stream":true,"messages":[{"role":"user","content":"Who is John?"}]}'
```

The reassembled text is:
`Sure! You can reach John at [REDACTED] or on 555-123-4567. His SSN is [REDACTED] and the card on file is [REDACTED]. ...`
(the phone number and the Luhn-invalid order number are intentionally left alone). `"stream": false` also works.

Settings: `PROVIDER_URL` (default `http://127.0.0.1:9003/v1/chat/completions`), `GATEWAY_PORT` (8003),
`MOCK_CHUNK_DELAY` for the mock provider (seconds, default 0.02).

### Task 4: Rate limiting and model fallback

```bash
# primary that hangs for 5s (modes: ok, rate_limited, slow, error) and a healthy secondary
uv run python -m task4_model_router.mock_providers --name primary --port 9004 --mode slow
uv run python -m task4_model_router.mock_providers --name secondary --port 9005
# gateway on :8004, SQLite file rate_limits.db in the current directory
uv run python -m task4_model_router
```

```bash
curl -si localhost:8004/v1/chat/completions -H 'Authorization: Bearer tenant-a-key' -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":200}'
# ~3s later: 200 with header x-gateway-provider: secondary
```

To see the limiter, restart the gateway with `TOKENS_PER_MINUTE=1000` and send three requests at once. Each reserves
409 tokens (5 prompt bytes + 4 overhead + 400 `max_tokens`) while the slow primary is timing out, so the third doesn't fit:

```bash
for i in 1 2 3; do curl -s -o /dev/null -w '%{http_code}\n' localhost:8004/v1/chat/completions \
  -H 'Authorization: Bearer tenant-a-key' -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":400}' & done; wait
# 200, 200, 429 (in some order)
```

Errors always look like `{"error": {"type": "rate_limit_exceeded", "message": "Token rate limit exceeded", "request_id": "req_..."}}`.

Settings: `PRIMARY_URL`, `SECONDARY_URL`, `PRIMARY_MODEL`, `SECONDARY_MODEL`, `PRIMARY_API_KEY`, `SECONDARY_API_KEY`,
`TENANT_API_KEYS` (comma separated, default `tenant-a-key,tenant-b-key`), `TOKENS_PER_MINUTE` (50000),
`RATE_LIMIT_DB` (`rate_limits.db`), `GATEWAY_PORT` (8004).

## Tests

```bash
uv run pytest                     # everything (~7s, one test deliberately waits for the real 3s timeout)
uv run pytest tests/test_task3.py # one task
```

What they cover, briefly:

- **Task 1**: valid calls, a table of bad customer IDs / amounts / reasons, missing and extra fields, unknown tool,
  `isError` vs JSON-RPC error, a raw subprocess session asserting every stdout line is JSON-RPC and logs land on
  stderr, -32700 for unparseable input, and the official SDK client talking to the server.
- **Task 2**: viewer vs admin, case tricks (`ADMIN_x`), 18 kinds of bad credentials (expired, wrong secret,
  `alg: none`, forged payload, non-string role, ...), malformed bodies and envelopes (bad id types, non-object
  params, `result` and `error` together), batches, duplicate-key smuggling, oversized streamed bodies, downstream
  down/timeout/broken mid-stream, SSE passthrough. Every rejection asserts the downstream received nothing.
- **Task 3**: detection against hand-written expected outputs (including cards and SSNs next to other digits),
  streaming output equal to whole-text redaction for many chunk sizes, 200 random chunkings and 2,000 seeded
  adversarial inputs, PII split at every kind of boundary, long-input speed and bounded buffer, first output
  emitted before the provider finishes, SSE format preserved, multiple choices, provider errors, RFC 5322 email
  local parts, the documented buffer bound, `logprobs`/refusal/tool-call fields never reaching the client,
  malformed provider events, and the provider response being closed on every exit path.
- **Task 4**: window boundaries with a fake clock, eviction from disk, persistence across instances, 40 threads and
  30 concurrent HTTP requests against one budget, fallback on 429/timeout/5xx/network/bad JSON, no fallback on 400,
  both providers failing, and error payloads that never contain upstream details.

## Design decisions

**Task 1.** Uses the SDK's low-level `Server` rather than `MCPServer`, because `MCPServer` turns argument validation
failures into `isError` tool results, and the assessment asks for JSON-RPC error codes. So:
- invalid arguments or unknown tool → JSON-RPC `-32602 Invalid params` with per-field details;
- valid request but the operation fails (unknown customer, refund above balance) → `result.isError = true`.

Pydantic models are `strict` (no `"10"` → 10 coercion) with `extra="forbid"`. The refund rules are exactly the
assessment's: `amount` is any finite positive number and `reason` has at least 10 characters, not counting surrounding
whitespace (so 20 spaces isn't a reason). Customer IDs use `[0-9]` instead of
`\d` because pydantic's regex engine treats `\d` as any Unicode digit. The SDK's stdio transport silently drops
lines it can't parse; a small filter in front of the server answers those with `-32700`/`-32600` as JSON-RPC
expects, echoing the request id when it can be read. The SDK also redirects fd 1 to stderr while serving, so even a stray `print` can't corrupt the protocol
stream, but all logging is configured for stderr regardless.

**Task 2.** Tokens are HS256 JWTs with a `role` claim; the signature, `exp`, `sub` and `role` are required and the
algorithm list is pinned, so the role can't be self-asserted. Other points:
- The gateway forwards the *re-serialized* parsed message, not the raw bytes. Otherwise a body with two `name`
  keys could be read one way by the gateway and another way by the downstream.
- Batches are rejected (MCP no longer supports them) so an admin call can't hide inside an array.
- `admin_` is matched case-insensitively after trimming whitespace.
- The envelope is checked before anything is forwarded: `jsonrpc` is `"2.0"`, ids are strings or integers,
  `params` is an object, and client responses carry exactly one of `result`/`error`. The 1 MB body limit is
  enforced while reading, so an oversized body is cut off rather than buffered.
- The downstream response is streamed through and always closed. If it breaks after headers were sent, the
  client connection is aborted rather than ended as if the body were complete.
- The caller's `Authorization` header is not forwarded downstream (no token passthrough).
- HTTP status: 401 for bad credentials, 400 for unparseable/invalid JSON-RPC, 200 with a JSON-RPC error for
  `-32001`/`-32602` (so MCP clients match it to the request id), 502/504 if the downstream is unreachable.

**Task 3.** `StreamRedactor` keeps a small pending buffer. After each chunk it holds back only the trailing run
of characters that could still become part of a match: email characters, or digits/spaces/dashes for SSNs and
cards. Everything before that is redacted and sent immediately, so the extra latency is roughly one word. The
hold-back is capped at 704 characters, the longest string any pattern can match, so text that has been sent can
never turn out to be the start of a match. The cut never goes through a match, so the buffer holds at most
704 + 703 = 1,407 characters per choice after each chunk. Tests check that streaming output is identical to
redacting the full text at once, including 2,000 seeded adversarial inputs. Digits are matched as whole runs (up to
48 digits) and then checked inside, so a card or SSN next to unrelated digits (`4111 1111 1111 1111 12/26`) is still
found. Email local parts accept the RFC 5322 characters (`o'connor`, `a!b`) but must start with a letter or digit, so
Markdown around an address (`**bob@x.com**`) is kept.

The SSE layer forwards an allowlist: per choice only `index`, `finish_reason` and a `delta` with `role` and the
redacted `content`, plus top-level `id`/`object`/`created`/`model`/`system_fingerprint`/`usage`. Everything else
(`logprobs`, tool calls, refusals, audio, unknown fields) is dropped, because it can repeat the generated text in
a form that isn't redacted; `logprobs` tokens and byte arrays would otherwise spell out a redacted email. The
non-streaming path applies the same allowlist to `message`. An event that isn't JSON or doesn't have that shape
ends the stream with a sanitized error event after flushing held text. Held text is flushed on `finish_reason`,
`[DONE]`, or end of stream, and the provider response is closed in a `finally` however the stream ends.

**Task 4.**
- Rate limiter is a sliding log: one SQLite row per request (`tenant`, `tokens`, `created_at`). The
  check-and-insert runs inside `BEGIN IMMEDIATE`, which takes SQLite's write lock up front, so concurrent requests
  (threads or separate processes on the same file) can't both see the same total and both pass. Rows older than
  the window are deleted on every reservation. Tenant keys are stored as SHA-256 prefixes.
- OpenAI-style requests may include `model`; it is accepted but ignored, since the router picks the provider model.
- Token accounting: before the call the gateway reserves `UTF-8 bytes of the prompt + 4 per message + max_tokens`
  (default `max_tokens` 512). After a successful call the reservation is replaced with the provider's
  `usage.total_tokens` (replaced, not added to, so completion tokens are never counted twice). If every provider
  fails the reservation is released, so tenants aren't charged for gateway-side failures.
- Why bytes: the byte-level BPE tokenizers most providers use emit at most one token per UTF-8 byte, so the byte
  count is an upper bound on prompt tokens. A `chars / 4` heuristic is close for English but several times too low
  for Chinese/Japanese text or emoji, and the tenant controls the prompt, so admission could be gamed. With the
  byte bound, admission never lets a request in on an underestimate of its prompt.
- Fallback: the primary call runs inside `asyncio.timeout(3.0)`, which covers connect, headers and body. httpx's
  own timeouts are per phase, so a slowly trickling response could otherwise run past 3s. When it fires, the
  primary request is cancelled before the secondary is called, so there is no race between a late primary
  response and the fallback. Fallback happens on 429, timeout, 5xx, network errors and unparseable responses,
  not on other 4xx.
- Errors: clients get `{"error": {"type", "message", "request_id"}}` with types `authentication_error` (401),
  `invalid_request` (400), `rate_limit_exceeded` (429 + `Retry-After`), `upstream_error` (502),
  `upstream_unavailable` (503), `internal_error` (500). Upstream bodies and exception text go to the logs only.

## Known limitations and tradeoffs

- **Not production-ready.** No TLS, no metrics, single-node SQLite, minimal configuration.
- **Task 1**: data is an in-memory dict, so refunds reset on restart and there is no idempotency key for retried
  refunds. `CUST-XXXXX` is interpreted as five digits.
- **Task 2**: a shared HMAC secret instead of an identity provider/JWKS; no token revocation. Only POST is proxied,
  so the streamable-HTTP GET (server-initiated SSE) and DELETE (session end) routes aren't. `tools/list` is
  forwarded unfiltered as the task requires, so viewers can see admin tool names; they just can't call them.
- **Task 3**: regex detection is pattern matching, not real PII detection. It covers emails, dashed US SSNs
  (`123-45-6789`) and 13-19 digit card numbers (optional single space/dash separators, Luhn-checked). It misses
  SSNs without dashes, obfuscated values (`john at example dot com`), numbers spelled out, PII in other languages
  or scripts, quoted or non-ASCII email local parts, cards glued to other digits with no separator
  (`41111111111111115500000000000004`), and cards straddling the boundary of a digit run longer than 48 digits.
  It can over-redact digits next to a card, numbers that happen to look like a dashed SSN or pass Luhn, and
  characters glued to the front of an email (`x=bob@x.com` loses `x=`). Only response text is passed through:
  clients asking for `logprobs` or tools get none back. The prompt isn't redacted. Assumes OpenAI-style
  single-line `data:` events; the 1,407-char bound is per choice, and the size of a single provider event line
  and the number of choices aren't capped.
- **Task 4**: the byte estimate is deliberately pessimistic, not a real tokenizer. English prompts reserve about
  4x their real token count (CJK about 2-3x) until the request settles, so a tenant can get 429s while actual usage
  is below the limit, and a request whose byte estimate exceeds the whole budget is rejected with 400 even if its
  real token count would fit (e.g. a 50 KB English prompt against a 50k tokens/min limit). The bound assumes a
  byte-level tokenizer and only allows 4 tokens per message for chat formatting; tokens a provider adds itself
  (hidden system prompts, tool definitions) aren't covered, so reported usage can still exceed the reservation,
  in which case the window briefly holds more than the limit and later requests are blocked until it drains.
  If a request is cancelled mid-flight its reservation stays at the estimate. One SQLite write lock serializes all reservations: fine for a single gateway node, not for a
  horizontally scaled fleet (use Redis or similar there). Only non-streaming completions are routed.
