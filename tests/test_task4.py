import asyncio
import json
import sqlite3
import threading
import time
from contextlib import closing

import httpx
import pytest

from task4_model_router.app import MAX_REPORTED_TOKENS, CompletionRequest, create_app, estimate_tokens, reported_tokens, tenant_id
from task4_model_router.mock_providers import create_provider_app
from task4_model_router.rate_limiter import RateLimitExceeded, TokenRateLimiter
from task4_model_router.router import ModelRouter, Provider, UpstreamError

PRIMARY = Provider("primary", "http://primary.test/v1/chat/completions", "primary-model")
SECONDARY = Provider("secondary", "http://secondary.test/v1/chat/completions", "backup-model")
TENANT_KEY = "tenant-a-key"
AUTH = {"Authorization": f"Bearer {TENANT_KEY}"}
PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "limits.db")


# ---------- rate limiter ----------


def test_admits_until_limit_then_rejects(db_path):
    limiter = TokenRateLimiter(db_path, limit_tokens=50_000, clock=FakeClock())
    limiter.reserve("t1", 30_000)
    limiter.reserve("t1", 20_000)

    with pytest.raises(RateLimitExceeded):
        limiter.reserve("t1", 1)
    assert limiter.tokens_used("t1") == 50_000


def test_sliding_window_boundaries_and_retry_after(db_path):
    clock = FakeClock()
    limiter = TokenRateLimiter(db_path, limit_tokens=50_000, window_seconds=60, clock=clock)
    start = clock.now
    limiter.reserve("t1", 30_000)
    clock.now = start + 30
    limiter.reserve("t1", 20_000)

    clock.now = start + 59.9
    with pytest.raises(RateLimitExceeded) as exc_info:
        limiter.reserve("t1", 10_000)
    # The 30k entry from t=0 expires at t=60.
    assert exc_info.value.retry_after_seconds == pytest.approx(0.1)

    clock.now = start + 60  # an entry exactly one window old is out of the window
    limiter.reserve("t1", 30_000)

    clock.now = start + 89.9
    with pytest.raises(RateLimitExceeded):
        limiter.reserve("t1", 1)
    clock.now = start + 90
    limiter.reserve("t1", 20_000)
    assert limiter.tokens_used("t1") == 50_000


def test_expired_rows_are_evicted_from_disk(db_path):
    clock = FakeClock()
    limiter = TokenRateLimiter(db_path, limit_tokens=50_000, clock=clock)
    for tenant in ["t1", "t2", "t3"]:
        limiter.reserve(tenant, 1_000)

    clock.now += 61
    limiter.reserve("t1", 500)

    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute("SELECT tenant, tokens FROM token_usage").fetchall()
    assert rows == [("t1", 500)]


def test_tenants_have_separate_budgets(db_path):
    limiter = TokenRateLimiter(db_path, limit_tokens=10_000, clock=FakeClock())
    limiter.reserve("t1", 10_000)
    limiter.reserve("t2", 10_000)
    with pytest.raises(RateLimitExceeded):
        limiter.reserve("t1", 1)


def test_settle_and_release_adjust_usage(db_path):
    limiter = TokenRateLimiter(db_path, limit_tokens=10_000, clock=FakeClock())
    first = limiter.reserve("t1", 6_000)
    limiter.settle(first, 1_200)
    second = limiter.reserve("t1", 8_000)
    assert limiter.tokens_used("t1") == 9_200

    limiter.release(second)
    assert limiter.tokens_used("t1") == 1_200


def test_usage_persists_across_limiter_instances(db_path):
    TokenRateLimiter(db_path, limit_tokens=10_000).reserve("t1", 9_000)

    reopened = TokenRateLimiter(db_path, limit_tokens=10_000)
    assert reopened.tokens_used("t1") == 9_000
    with pytest.raises(RateLimitExceeded):
        reopened.reserve("t1", 2_000)


def test_concurrent_threads_cannot_exceed_budget(db_path):
    limiter = TokenRateLimiter(db_path, limit_tokens=50_000)
    results = []
    barrier = threading.Barrier(40)

    def worker():
        barrier.wait()
        try:
            limiter.reserve("t1", 5_000)
            results.append("ok")
        except RateLimitExceeded:
            results.append("limited")

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count("ok") == 10
    assert limiter.tokens_used("t1") == 50_000


@pytest.mark.parametrize(
    "messages, prompt_bytes",
    [
        ([{"role": "user", "content": "Hello, world!"}], 13),  # English: 1 byte per char
        ([{"role": "user", "content": "你好，世界"}], 15),  # Chinese: 3 bytes per char
        ([{"role": "user", "content": "こんにちは"}], 15),  # Japanese: 3 bytes per char
        ([{"role": "user", "content": "👍🏽"}], 8),  # emoji + skin tone modifier: 4 bytes each
        ([{"role": "user", "content": "café e\u0301"}], 9),  # precomposed é (2) and e + combining accent (1 + 2)
        ([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "日本"}], 9 + 6),
    ],
)
def test_token_estimate_reserves_utf8_bytes_plus_overhead_and_completion(messages, prompt_bytes):
    request = CompletionRequest.model_validate({"messages": messages, "max_tokens": 50})
    assert estimate_tokens(request) == prompt_bytes + 4 * len(messages) + 50


def test_token_estimate_is_at_least_one_token_per_character_for_cjk():
    # chars/4 would reserve 25 tokens here; real tokenizers use roughly 100 or more.
    request = CompletionRequest.model_validate({"messages": [{"role": "user", "content": "漢" * 100}], "max_tokens": 1})
    assert estimate_tokens(request) >= 100


def test_admission_uses_the_time_the_sqlite_lock_was_acquired(db_path):
    # A request that starts waiting for the write lock at t=159.9 and gets it at t=160.1 must be judged
    # at 160.1, when the t=100 entry has left the 60s window, and its row must be stamped 160.1.
    clock = FakeClock(100.0)
    limiter = TokenRateLimiter(db_path, limit_tokens=100, window_seconds=60, clock=clock)
    limiter.reserve("t1", 100)

    clock.now = 159.9
    holder = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")

    def release_lock_later():
        time.sleep(0.3)
        clock.now = 160.1
        holder.execute("COMMIT")

    releaser = threading.Thread(target=release_lock_later)
    releaser.start()
    try:
        limiter.reserve("t1", 50)
    finally:
        releaser.join()
        holder.close()

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT tokens, created_at FROM token_usage").fetchall() == [(50, 160.1)]


# ---------- router ----------


def completion(text="hi", total_tokens=42):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}], "usage": {"total_tokens": total_tokens}}


class ProviderMock:
    """Routes by host to per-provider behaviors and records calls."""

    def __init__(self, primary="ok", secondary="ok", slow_seconds=5.0, total_tokens=42):
        self.behaviors = {"primary.test": primary, "secondary.test": secondary}
        self.slow_seconds = slow_seconds
        self.total_tokens = total_tokens
        self.calls: list[str] = []

    async def __call__(self, request: httpx.Request):
        host = request.url.host
        self.calls.append(host.split(".")[0])
        behavior = self.behaviors[host]
        if behavior == "ok":
            return httpx.Response(200, json=completion(f"from {host}", self.total_tokens))
        if behavior == "429":
            return httpx.Response(429, json={"error": "slow down"})
        if behavior == "500":
            return httpx.Response(500, text="Traceback (most recent call last): internal db password=hunter2")
        if behavior == "400":
            return httpx.Response(400, json={"error": "bad request"})
        if behavior == "bad_json":
            return httpx.Response(200, text="<html>oops</html>")
        if behavior == "too_deep":  # parsing raises RecursionError rather than ValueError
            return httpx.Response(200, text="[" * 100_000 + "]" * 100_000)
        if behavior == "slow":
            await asyncio.sleep(self.slow_seconds)
            return httpx.Response(200, json=completion("too late", self.total_tokens))
        if behavior == "connect_error":
            raise httpx.ConnectError("connection refused")
        raise AssertionError(behavior)


def make_router(mock, primary_timeout=0.2, secondary_timeout=0.5):
    client = httpx.AsyncClient(transport=httpx.MockTransport(mock))
    return ModelRouter(PRIMARY, SECONDARY, client, primary_timeout=primary_timeout, secondary_timeout=secondary_timeout)


async def test_primary_success_does_not_touch_secondary():
    mock = ProviderMock()
    result, provider = await make_router(mock).complete(PAYLOAD)
    assert provider == "primary"
    assert mock.calls == ["primary"]


@pytest.mark.parametrize("failure", ["429", "slow", "500", "connect_error", "bad_json", "too_deep"])
async def test_falls_back_to_secondary(failure):
    mock = ProviderMock(primary=failure)
    started = time.perf_counter()
    result, provider = await make_router(mock).complete(PAYLOAD)

    assert provider == "secondary"
    assert result["choices"][0]["message"]["content"] == "from secondary.test"
    assert mock.calls == ["primary", "secondary"]
    assert time.perf_counter() - started < 1.0


async def test_default_primary_timeout_is_3000ms():
    mock = ProviderMock(primary="slow", slow_seconds=3.5)
    router = ModelRouter(PRIMARY, SECONDARY, httpx.AsyncClient(transport=httpx.MockTransport(mock)))
    started = time.perf_counter()
    _, provider = await router.complete(PAYLOAD)
    elapsed = time.perf_counter() - started

    assert provider == "secondary"
    assert 3.0 <= elapsed < 3.4


async def test_primary_answering_just_before_deadline_wins():
    mock = ProviderMock(primary="slow", slow_seconds=0.1)
    _, provider = await make_router(mock, primary_timeout=0.3).complete(PAYLOAD)
    assert provider == "primary"
    assert mock.calls == ["primary"]


async def test_non_retryable_rejection_does_not_fall_back():
    mock = ProviderMock(primary="400")
    with pytest.raises(UpstreamError) as exc_info:
        await make_router(mock).complete(PAYLOAD)
    assert exc_info.value.retryable is False
    assert mock.calls == ["primary"]


@pytest.mark.parametrize("secondary", ["429", "slow", "500"])
async def test_both_providers_failing_raises(secondary):
    with pytest.raises(UpstreamError) as exc_info:
        await make_router(ProviderMock(primary="429", secondary=secondary)).complete(PAYLOAD)
    assert exc_info.value.provider == "secondary"


async def test_sends_provider_specific_model_name():
    seen = []

    def handler(request):
        seen.append(request.read())
        return httpx.Response(200, json=completion())

    router = ModelRouter(PRIMARY, SECONDARY, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await router.complete(PAYLOAD)
    assert b'"model":"primary-model"' in seen[0].replace(b" ", b"")


# ---------- HTTP gateway ----------


def gateway(db_path, mock, limit_tokens=50_000, clock=None):
    limiter = TokenRateLimiter(db_path, limit_tokens=limit_tokens, clock=clock or time.time)
    app = create_app(limiter, make_router(mock), {TENANT_KEY, "tenant-b-key"})
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway"), limiter


def assert_gateway_error(response, status, error_type):
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "message", "request_id"}
    assert body["error"]["type"] == error_type
    assert body["error"]["request_id"] == response.headers["x-request-id"]


async def test_successful_request_reports_provider_and_settles_usage(db_path):
    client, limiter = gateway(db_path, ProviderMock(primary="429"))
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)

    assert response.status_code == 200
    assert response.headers["x-gateway-provider"] == "secondary"
    # Reservation (estimate) replaced by the provider's reported total_tokens.
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 42


async def test_openai_style_model_field_is_accepted_but_gateway_picks_model(db_path):
    sent_models = []

    def handler(request):
        sent_models.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=completion())

    limiter = TokenRateLimiter(db_path)
    router = ModelRouter(PRIMARY, SECONDARY, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    app = create_app(limiter, router, {TENANT_KEY})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post("/v1/chat/completions", json={**PAYLOAD, "model": "gpt-4o"}, headers=AUTH)

    assert response.status_code == 200
    assert sent_models == ["primary-model"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TENANT_KEY}])
async def test_rejects_unknown_api_keys(db_path, headers):
    mock = ProviderMock()
    client, _ = gateway(db_path, mock)
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=headers)
    assert_gateway_error(response, 401, "authentication_error")
    assert mock.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "hacker", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": -5},
        {"messages": [{"role": "user", "content": "x"}], "unexpected": True},
        "not an object",
    ],
)
async def test_invalid_bodies_return_standard_error(db_path, body):
    client, _ = gateway(db_path, ProviderMock())
    async with client:
        response = await client.post("/v1/chat/completions", json=body, headers=AUTH)
    assert_gateway_error(response, 400, "invalid_request")
    assert "pydantic" not in response.text


async def test_request_larger_than_budget_is_rejected(db_path):
    client, _ = gateway(db_path, ProviderMock(), limit_tokens=1_000)
    async with client:
        response = await client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 2_000}, headers=AUTH)
    assert_gateway_error(response, 400, "invalid_request")


async def test_rate_limited_tenant_gets_429_with_retry_after(db_path):
    mock = ProviderMock()
    client, limiter = gateway(db_path, mock, limit_tokens=1_000)
    limiter.reserve(tenant_id(TENANT_KEY), 950)
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)
        other_tenant = await client.post(
            "/v1/chat/completions", json=PAYLOAD, headers={"Authorization": "Bearer tenant-b-key"}
        )

    assert_gateway_error(response, 429, "rate_limit_exceeded")
    assert 1 <= int(response.headers["retry-after"]) <= 60
    assert other_tenant.status_code == 200
    assert mock.calls == ["primary"]


async def test_concurrent_requests_cannot_bypass_budget(db_path):
    # Each request reserves 2 bytes + 4 + 994 = 1000 tokens against a 10k budget. The provider
    # reports the same total, so the result doesn't depend on when reservations get settled.
    mock = ProviderMock(primary="slow", slow_seconds=0.05, total_tokens=1000)
    client, limiter = gateway(db_path, mock, limit_tokens=10_000)
    payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 994}
    async with client:
        responses = await asyncio.gather(
            *[client.post("/v1/chat/completions", json=payload, headers=AUTH) for _ in range(30)]
        )

    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 10
    assert statuses.count(429) == 20
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 10_000


async def test_concurrent_multibyte_requests_near_budget(db_path):
    # "日本語" is 9 bytes, so each request reserves 9 + 4 + 87 = 100 tokens of a 1,000 budget.
    mock = ProviderMock(primary="slow", slow_seconds=0.05, total_tokens=100)
    client, limiter = gateway(db_path, mock, limit_tokens=1_000)
    payload = {"messages": [{"role": "user", "content": "日本語"}], "max_tokens": 87}
    async with client:
        responses = await asyncio.gather(
            *[client.post("/v1/chat/completions", json=payload, headers=AUTH) for _ in range(12)]
        )

    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 10
    assert statuses.count(429) == 2
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 1_000


async def send_sequentially(client, count):
    statuses = []
    for _ in range(count):
        response = await client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 400}, headers=AUTH)
        statuses.append(response.status_code)
    return statuses


async def test_lower_reported_usage_frees_budget(db_path):
    # Each request reserves 5 + 4 + 400 = 409. Without reconciliation the third wouldn't fit in 1,000.
    client, limiter = gateway(db_path, ProviderMock(total_tokens=50), limit_tokens=1_000)
    async with client:
        assert await send_sequentially(client, 3) == [200, 200, 200]
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 150


async def test_higher_reported_usage_is_charged_and_blocks_next_request(db_path):
    client, limiter = gateway(db_path, ProviderMock(total_tokens=900), limit_tokens=1_000)
    async with client:
        assert await send_sequentially(client, 2) == [200, 429]
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 900


async def test_missing_reported_usage_keeps_the_estimate(db_path):
    client, limiter = gateway(db_path, ProviderMock(total_tokens=None), limit_tokens=1_000)
    async with client:
        assert await send_sequentially(client, 1) == [200]
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 409


def provider_reporting(usage):
    def handler(request):
        body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
        if usage != "absent":
            body["usage"] = usage
        return httpx.Response(200, json=body)

    return handler


@pytest.mark.parametrize(
    "usage, charged",
    [
        ({"total_tokens": True}, 409),  # bool is not a count; keep the estimate
        ({"total_tokens": False}, 409),
        ({"total_tokens": 12.0}, 409),
        ({"total_tokens": "12"}, 409),
        ({"total_tokens": -5}, 409),
        (["bad"], 409),
        ("12", 409),
        (None, 409),
        ("absent", 409),
        ({"total_tokens": 2**70}, 10**12),  # capped so it fits in (and sums within) SQLite's 64-bit integers
        ({"total_tokens": 5_000}, 5_000),  # above the budget, stored as reported
        ({"total_tokens": 0}, 0),
        ({"total_tokens": 12}, 12),
    ],
)
async def test_reported_usage_is_only_trusted_when_it_is_a_sane_integer(db_path, usage, charged):
    limiter = TokenRateLimiter(db_path, limit_tokens=1_000)
    router = ModelRouter(PRIMARY, SECONDARY, httpx.AsyncClient(transport=httpx.MockTransport(provider_reporting(usage))))
    app = create_app(limiter, router, {TENANT_KEY})
    async with router.client, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        response = await client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 400}, headers=AUTH)
        assert limiter.tokens_used(tenant_id(TENANT_KEY)) == charged
        # A 10-token request (5 prompt bytes + 4 + 1) fits unless the first one used the whole budget.
        follow_up = await client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 1}, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    assert follow_up.status_code == (429 if charged > 990 else 200)


async def test_usage_above_the_budget_still_counts_after_the_budget_is_raised(db_path):
    clock = FakeClock()
    router = ModelRouter(PRIMARY, SECONDARY, httpx.AsyncClient(transport=httpx.MockTransport(provider_reporting({"total_tokens": 150}))))
    app = create_app(TokenRateLimiter(db_path, limit_tokens=100, clock=clock), router, {TENANT_KEY})
    async with router.client, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        first = await client.post("/v1/chat/completions", json={**PAYLOAD, "max_tokens": 10}, headers=AUTH)
    assert first.status_code == 200

    # Restart with a higher budget a second later; the 150 tokens used are still in the window.
    clock.now += 1
    raised = TokenRateLimiter(db_path, limit_tokens=200, clock=clock)
    assert raised.tokens_used(tenant_id(TENANT_KEY)) == 150
    with pytest.raises(RateLimitExceeded):
        raised.reserve(tenant_id(TENANT_KEY), 75)
    raised.reserve(tenant_id(TENANT_KEY), 50)
    assert raised.tokens_used(tenant_id(TENANT_KEY)) == 200


@pytest.mark.parametrize("total", [2**31, 2**63 - 1, 2**63, 2**70, 10**400], ids=["2^31", "2^63-1", "2^63", "2^70", "10^400"])
def test_huge_reported_usage_is_stored_without_overflowing_sqlite(db_path, total):
    assert reported_tokens({"usage": {"total_tokens": total}}) == min(total, MAX_REPORTED_TOKENS)
    limiter = TokenRateLimiter(db_path, limit_tokens=50_000)
    # Several in-flight requests settling huge counts must still sum within 64-bit integers.
    reservations = [limiter.reserve("t1", 100) for _ in range(5)]
    for reservation in reservations:
        limiter.settle(reservation, reported_tokens({"usage": {"total_tokens": total}}))

    assert limiter.tokens_used("t1") == 5 * min(total, 10**12)
    with pytest.raises(RateLimitExceeded):
        limiter.reserve("t1", 1)


async def test_router_client_is_closed_when_the_app_shuts_down(db_path):
    router = make_router(ProviderMock())
    app = create_app(TokenRateLimiter(db_path), router, {TENANT_KEY})
    async with app.router.lifespan_context(app):
        assert not router.client.is_closed
    assert router.client.is_closed


async def test_router_client_is_closed_when_the_lifespan_exits_with_an_error(db_path):
    router = make_router(ProviderMock())
    app = create_app(TokenRateLimiter(db_path), router, {TENANT_KEY})
    with pytest.raises(RuntimeError):
        async with app.router.lifespan_context(app):
            raise RuntimeError("startup failed")
    assert router.client.is_closed


async def test_multibyte_prompt_can_be_rejected_even_if_real_tokens_would_fit(db_path):
    # 300 CJK chars = 900 bytes; with max_tokens 100 that reserves 1,004 > 1,000, although a real
    # tokenizer would likely count ~300-450 prompt tokens. Documented tradeoff of the byte estimate.
    client, _ = gateway(db_path, ProviderMock(), limit_tokens=1_000)
    payload = {"messages": [{"role": "user", "content": "你" * 300}], "max_tokens": 100}
    async with client:
        response = await client.post("/v1/chat/completions", json=payload, headers=AUTH)
    assert_gateway_error(response, 400, "invalid_request")


async def test_upstream_failures_are_sanitized_and_not_charged(db_path):
    client, limiter = gateway(db_path, ProviderMock(primary="500", secondary="500"))
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)

    assert_gateway_error(response, 503, "upstream_unavailable")
    for leaked in ["Traceback", "hunter2", "primary.test", "secondary.test", "500"]:
        assert leaked not in response.json()["error"]["message"]
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == 0


@pytest.mark.parametrize("secondary, status, provider, used", [("ok", 200, "secondary", 42), ("too_deep", 503, None, 0)])
async def test_too_deeply_nested_provider_json_falls_back_and_is_accounted(db_path, secondary, status, provider, used):
    client, limiter = gateway(db_path, ProviderMock(primary="too_deep", secondary=secondary))
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)

    assert response.status_code == status
    assert response.headers.get("x-gateway-provider") == provider
    if status == 503:
        assert_gateway_error(response, 503, "upstream_unavailable")
    # Settled to the secondary's reported usage, or released when no provider answered.
    assert limiter.tokens_used(tenant_id(TENANT_KEY)) == used


async def test_provider_rejection_maps_to_502(db_path):
    client, _ = gateway(db_path, ProviderMock(primary="400"))
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)
    assert_gateway_error(response, 502, "upstream_error")


async def test_unexpected_errors_do_not_leak_internals(db_path, monkeypatch):
    client, limiter = gateway(db_path, ProviderMock())

    def broken_reserve(*args):
        raise RuntimeError("disk I/O error at /var/lib/gateway/limits.db")

    monkeypatch.setattr(limiter, "reserve", broken_reserve)
    async with client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)

    assert_gateway_error(response, 500, "internal_error")
    assert "/var/lib" not in response.text
    assert "RuntimeError" not in response.text


async def test_end_to_end_with_mock_provider_apps(db_path):
    primary_app = create_provider_app("primary", mode="rate_limited")
    secondary_app = create_provider_app("secondary")

    async def route(request: httpx.Request):
        app = primary_app if request.url.host == "primary.test" else secondary_app
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
            return await c.post(request.url.path, content=request.content, headers={"content-type": "application/json"})

    transport = httpx.MockTransport(route)
    limiter = TokenRateLimiter(db_path)
    router = ModelRouter(PRIMARY, SECONDARY, httpx.AsyncClient(transport=transport))
    app = create_app(limiter, router, {TENANT_KEY})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        response = await client.post("/v1/chat/completions", json=PAYLOAD, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello from secondary."
    assert response.json()["model"] == "backup-model"
