import asyncio
import json
import random
import time

import httpx
import pytest

from task3_streaming_guardrail import mock_provider
from task3_streaming_guardrail.gateway import create_app, redact_sse_stream
from task3_streaming_guardrail.redactor import MAX_HOLDBACK, StreamRedactor, luhn_valid, redact_text

SAMPLES = [
    "Contact john.doe@example.com for details.",
    "SSN: 123-45-6789, card 4111 1111 1111 1111, mail a.b+tag@mail.co.uk!",
    "Cards: 4111-1111-1111-1111 and 5500000000000004 and 378282246310005.",
    "Email me at x@y.io\nor jane_smith99@sub.domain.org.",
    "Order 1234567890123 shipped; call 555-123-4567 tomorrow.",
    "Version 2.0.1 released on 2024-01-15 with 42 fixes.",
    "123-45-6789123-45-6789 is not an SSN, but 987-65-4321 is.",
    "trailing pii at the end alice@example.com",
    "ends with a card 4242424242424242",
    "",
]


def stream_through(text: str, chunks: list[str]) -> str:
    redactor = StreamRedactor()
    return "".join(redactor.feed(chunk) for chunk in chunks) + redactor.flush()


def split_every(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def test_redact_text_patterns():
    assert redact_text("mail john.doe@example.com now") == "mail [REDACTED] now"
    assert redact_text("ssn 123-45-6789.") == "ssn [REDACTED]."
    assert redact_text("card 4111 1111 1111 1111 ok") == "card [REDACTED] ok"
    assert redact_text("card 4111-1111-1111-1111") == "card [REDACTED]"
    assert redact_text("amex 378282246310005") == "amex [REDACTED]"


def test_non_sensitive_text_is_unchanged():
    text = "Order 1234567890123 shipped; call 555-123-4567. Version 2.0.1, user@ handle, 12-34-5678."
    assert redact_text(text) == text


def test_luhn_filters_random_long_numbers():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert redact_text("id 4111111111111112") == "id 4111111111111112"


PII_NEXT_TO_DIGITS = [
    ("Card 4111 1111 1111 1111 12/26", "Card [REDACTED] 12/26"),
    ("Card 4111 1111 1111 1111 123", "Card [REDACTED] 123"),
    ("Ref 12 4111 1111 1111 1111", "Ref 12 [REDACTED]"),
    ("Phone 555 123 4567 123-45-6789", "Phone 555 123 4567 [REDACTED]"),
    ("Codes 1 2 3 4 123-45-6789", "Codes 1 2 3 4 [REDACTED]"),
    ("Ext 12 34 123-45-6789", "Ext 12 34 [REDACTED]"),
    ("4111 1111 1111 1111 123-45-6789", "[REDACTED] [REDACTED]"),
    ("amex 3782 822463 10005 exp 0127", "amex [REDACTED] exp 0127"),
    ("Pay with 4111111111111111 or 5500-0000-0000-0004 now", "Pay with [REDACTED] or [REDACTED] now"),
    ("Dates 2024 01 15 2024 02 20, call 555 123 4567", "Dates 2024 01 15 2024 02 20, call 555 123 4567"),
]


@pytest.mark.parametrize("text, expected", PII_NEXT_TO_DIGITS)
def test_pii_next_to_other_digits(text, expected):
    assert redact_text(text) == expected


@pytest.mark.parametrize("text, expected", PII_NEXT_TO_DIGITS)
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_pii_next_to_other_digits_when_streamed(text, expected, chunk_size):
    assert stream_through(text, split_every(text, chunk_size)) == expected


def test_card_at_end_of_long_digit_run_is_redacted():
    # 23 digits in total; the card is the last four groups. Extra digits may be redacted too.
    output = redact_text("Codes 1 2 3 4 5 6 7 4111 1111 1111 1111 done")
    assert output.startswith("Codes 1 2 ")
    assert output.endswith("[REDACTED] done")
    assert "4111" not in output and "1111" not in output


@pytest.mark.parametrize("text", SAMPLES)
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 16, 1000])
def test_streaming_matches_whole_text_redaction(text, chunk_size):
    assert stream_through(text, split_every(text, chunk_size)) == redact_text(text)


@pytest.mark.parametrize("seed", range(200))
def test_random_chunk_boundaries_match_whole_text_redaction(seed):
    rng = random.Random(seed)
    words = ["hello", "world", "a@b.com", "x.y@corp.example.org", "123-45-6789", "4111 1111 1111 1111",
             "5500-0000-0000-0004", "555-123-4567", "2024", "-", "foo@", "@bar", "4111111111111112", "42"]
    text = rng.choice([" ", ", ", "\n"]).join(rng.choice(words) for _ in range(rng.randint(1, 30)))
    chunks, position = [], 0
    while position < len(text):
        size = rng.randint(1, 12)
        chunks.append(text[position : position + size])
        position += size
    assert stream_through(text, chunks) == redact_text(text)


def test_adversarial_inputs_stream_the_same_as_whole_text():
    # Glued tokens with no separators, stray @ and -, PII next to digits, non-ASCII digits.
    tokens = ["a", "Z", "0", "7", "-", " ", ".", "@", "_", "+", "x@y.co", "123-45-6789", "4111 1111 1111 1111",
              "4111111111111111", "b@c.org", "\n", ",", "foo", "99", "4242-4242-4242-4242", "é", "١", "12/26",
              "555 123 4567", "'", "!", "*", "`", "=", "o'c", "{x}", "~", "#"]
    failures = []
    for seed in range(2000):
        rng = random.Random(seed)
        text = rng.choice(["", " "]).join(rng.choice(tokens) for _ in range(rng.randint(1, 25)))
        chunks, position = [], 0
        while position < len(text):
            size = rng.randint(1, 8)
            chunks.append(text[position : position + size])
            position += size
        if stream_through(text, chunks) != redact_text(text):
            failures.append((seed, text))
    assert failures == []


def test_long_inputs_stay_fast_and_bounded():
    started = time.perf_counter()
    redact_text("a" * 200_000)
    redact_text("x@" + "ab." * 50_000)
    redact_text("4 " * 50_000)
    assert time.perf_counter() - started < 1.0

    # Email-like run that never ends: the buffer must not grow with it.
    redactor = StreamRedactor()
    for chunk in split_every("x@" + "ab." * 20_000, 5):
        redactor.feed(chunk)
        assert len(redactor.pending) <= 1_000


EMAIL_LOCAL_PARTS = [
    ("Mail o'connor@example.com now", "Mail [REDACTED] now"),
    ("Mail a!b@example.com now", "Mail [REDACTED] now"),
    ("Mail j.o'neil+promo@mail.example.co.uk.", "Mail [REDACTED]."),
    ("Mail x#y$z%w&v*u/t=s?r^q_p`o{n|m}l~k-j@example.org", "Mail [REDACTED]"),
    ("Bold **bob@example.com** and code `bob@example.com`", "Bold **[REDACTED]** and code `[REDACTED]`"),
    ("Quoted 'bob@example.com' or (bob@example.com)", "Quoted '[REDACTED]' or ([REDACTED])"),
    ("Not an email: it's me! @home, a@b, x@y.z", "Not an email: it's me! @home, a@b, x@y.z"),
]


@pytest.mark.parametrize("text, expected", EMAIL_LOCAL_PARTS)
def test_whole_email_local_part_is_redacted(text, expected):
    assert redact_text(text) == expected
    for size in (1, 2, 5):
        assert stream_through(text, split_every(text, size)) == expected


def test_emails_longer_than_the_old_256_char_cap_are_not_released_early():
    # Six 63-digit labels: a 400-char string the email pattern accepts once ".com" arrives.
    email = "bob@" + ".".join(str(digit) * 63 for digit in range(1, 7)) + ".com"
    text = f"mail {email} now"
    assert redact_text(text) == "mail [REDACTED] now"
    assert stream_through(text, split_every(text, 1)) == "mail [REDACTED] now"


def test_pending_buffer_bound_matches_documented_limit():
    # Longest email the pattern allows: 64 + 1 + 63 + 8 * 64 + 1 + 63 = 704 chars. The buffer may hold
    # one held-back window plus one match crossing it: 704 + 703 = 1,407 chars after any feed.
    longest = "a" * 64 + "@" + ".".join(["b" * 63] * 9) + "." + "c" * 63
    assert len(longest) == 704 == MAX_HOLDBACK
    assert redact_text(f"<{longest}>") == "<[REDACTED]>"
    peak = 0
    redactor = StreamRedactor()
    emitted = []
    for text in ["z" * 703 + longest, longest + "d" * 2000, ("x" * 63 + longest) * 3]:
        for chunk in split_every(text, 1):
            emitted.append(redactor.feed(chunk))
            peak = max(peak, len(redactor.pending))
    emitted.append(redactor.flush())
    assert peak <= 1407
    assert "b" * 63 not in "".join(emitted)


@pytest.mark.parametrize(
    "chunks",
    [
        ["Reach me at john", ".doe@", "exam", "ple.com today"],
        ["SSN 123-4", "5-67", "89 on file"],
        ["card 4111 11", "11 1111 ", "1111", " thanks"],
        ["x", "@", "y", ".", "c", "o", "m", " "],
    ],
)
def test_pii_split_across_chunks_never_leaks(chunks):
    redactor = StreamRedactor()
    emitted = [redactor.feed(chunk) for chunk in chunks] + [redactor.flush()]
    output = "".join(emitted)

    assert "[REDACTED]" in output
    for secret_part in ["john.doe", "example.com", "6789", "4111", "y.com"]:
        assert secret_part not in output


def test_safe_text_is_emitted_without_waiting_for_end_of_stream():
    redactor = StreamRedactor()
    assert redactor.feed("The quick brown fox ") == "The quick brown fox"
    assert redactor.feed("jumps over") == " jumps "
    assert redactor.pending == "over"


def test_memory_stays_bounded_on_long_runs():
    redactor = StreamRedactor()
    emitted = []
    for text in ["a" * 5000, "7" * 5000, "word " * 2000]:
        for chunk in split_every(text, 37):
            emitted.append(redactor.feed(chunk))
            assert len(redactor.pending) <= MAX_HOLDBACK + 37
    emitted.append(redactor.flush())
    assert len("".join(emitted)) == 5000 + 5000 + 5 * 2000


def sse_lines(chunks: list[str], extra: dict | None = None) -> list[str]:
    lines = []
    for chunk in chunks:
        event = {"id": "c1", "model": "m", "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
        lines += [f"data: {json.dumps(event)}", ""]
    lines += [f"data: {json.dumps({'id': 'c1', 'model': 'm', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}", ""]
    lines += ["data: [DONE]", ""]
    return lines


def parse_sse(raw: bytes) -> tuple[list[dict], bool]:
    events, done = [], False
    for block in raw.decode().split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: "), block
        payload = block[len("data: ") :]
        if payload == "[DONE]":
            done = True
        else:
            events.append(json.loads(payload))
    return events, done


def content_of(events: list[dict]) -> str:
    return "".join(c["delta"].get("content") or "" for e in events for c in e.get("choices", []))


async def collect(async_iterable) -> bytes:
    return b"".join([item async for item in async_iterable])


async def as_async(lines):
    for line in lines:
        yield line


async def test_sse_stream_is_redacted_and_well_formed():
    text = "Hi, email bob@example.com or SSN 123-45-6789."
    raw = await collect(redact_sse_stream(as_async(sse_lines(split_every(text, 3)))))
    events, done = parse_sse(raw)

    assert done
    assert content_of(events) == "Hi, email [REDACTED] or SSN [REDACTED]."
    assert all(event["id"] == "c1" and event["model"] == "m" for event in events)
    assert events[-1]["choices"][0]["finish_reason"] == "stop"


async def test_sse_held_text_is_flushed_when_provider_omits_finish_reason():
    lines = [f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': 'mail a@b.com'}}]})}", "data: [DONE]"]
    events, done = parse_sse(await collect(redact_sse_stream(as_async(lines))))
    assert done
    assert content_of(events) == "mail [REDACTED]"


async def test_sse_choices_are_redacted_independently():
    def event(index, content):
        return f"data: {json.dumps({'choices': [{'index': index, 'delta': {'content': content}}]})}"

    lines = [event(0, "a@exa"), event(1, "x 123-"), event(0, "mple.com"), event(1, "45-6789 y"), "data: [DONE]"]
    events, _ = parse_sse(await collect(redact_sse_stream(as_async(lines))))
    by_choice = {0: "", 1: ""}
    for e in events:
        for choice in e["choices"]:
            by_choice[choice["index"]] += choice["delta"].get("content") or ""
    assert by_choice == {0: "[REDACTED]", 1: "x [REDACTED] y"}


async def test_output_streams_before_provider_finishes():
    release_rest = asyncio.Event()

    async def slow_provider():
        for line in sse_lines(["Hello there, "]):
            yield line
        await release_rest.wait()
        for line in sse_lines(["goodbye."]):
            yield line

    stream = redact_sse_stream(slow_provider())
    first = await asyncio.wait_for(anext(stream), timeout=1)
    assert "Hello there," in first.decode()

    release_rest.set()
    await collect(stream)


def provider_transport(chunks: list[str], status: int = 200):
    async def body():
        for line in sse_lines(chunks):
            yield (line + "\n").encode()

    def handler(request):
        if status != 200:
            return httpx.Response(status, json={"error": "Traceback (most recent call last): secret internals"})
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def gateway_client(transport) -> httpx.AsyncClient:
    app = create_app("http://provider.test/v1/chat/completions", http_client=httpx.AsyncClient(transport=transport))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")


REQUEST = {"model": "mock", "stream": True, "messages": [{"role": "user", "content": "hi"}]}


async def test_endpoint_streams_redacted_sse():
    text = "Card 4111 1111 1111 1111 and mail z@q.dev done."
    async with gateway_client(provider_transport(split_every(text, 4))) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events, done = parse_sse(response.content)
    assert done
    assert content_of(events) == "Card [REDACTED] and mail [REDACTED] done."


async def test_endpoint_with_mock_provider_app():
    async with gateway_client(httpx.ASGITransport(app=mock_provider.app)) as client:
        streamed = await client.post("/v1/chat/completions", json=REQUEST)
        plain = await client.post("/v1/chat/completions", json={**REQUEST, "stream": False})

    events, done = parse_sse(streamed.content)
    expected = redact_text(mock_provider.RESPONSE_TEXT)
    assert done
    assert content_of(events) == expected
    assert plain.json()["choices"][0]["message"]["content"] == expected
    assert "john.doe@example.com" not in streamed.text + plain.text


async def test_provider_errors_are_sanitized():
    async with gateway_client(provider_transport([], status=500)) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == 502
    assert response.json() == {"error": {"type": "upstream_error", "message": "LLM provider returned an error"}}

    def refuse(request):
        raise httpx.ConnectError("connection refused to 10.0.0.5")

    async with gateway_client(httpx.MockTransport(refuse)) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == 502
    assert "10.0.0.5" not in response.text


async def test_invalid_requests_are_rejected():
    async with gateway_client(provider_transport([])) as client:
        bad_json = await client.post("/v1/chat/completions", content=b"{nope")
        no_messages = await client.post("/v1/chat/completions", json={"model": "m"})
    assert bad_json.status_code == 400
    assert no_messages.status_code == 400


def event_line(choice: dict, **fields) -> str:
    return f"data: {json.dumps({'id': 'c1', 'model': 'm', **fields, 'choices': [choice]})}"


def logprob_choice(token: str) -> dict:
    alternative = {"token": token, "bytes": list(token.encode()), "logprob": -0.5}
    return {
        "index": 0,
        "delta": {"content": token, "refusal": token, "tool_calls": [{"function": {"arguments": token}}]},
        "logprobs": {"content": [{**alternative, "top_logprobs": [alternative]}], "refusal": None},
        "finish_reason": None,
    }


async def test_only_redacted_content_leaves_the_stream():
    # The email arrives split across tokens; logprobs, refusals and tool-call arguments repeat each token.
    tokens = ["Mail ", "bob", "@", "example", ".com", " ok"]
    lines = [event_line(logprob_choice(token), debug="bob@example.com") for token in tokens]
    lines += [event_line({"index": 0, "delta": {}, "finish_reason": "stop"}, usage={"total_tokens": 9}), "data: [DONE]"]

    raw = await collect(redact_sse_stream(as_async(lines)))
    events, done = parse_sse(raw)

    assert done
    assert content_of(events) == "Mail [REDACTED] ok"
    for secret in [b"bob", b"example", b"98, 111, 98", b"logprobs", b"refusal", b"tool_calls", b"debug"]:
        assert secret not in raw
    for event in events:
        assert set(event) <= {"id", "model", "usage", "choices"}
        for choice in event["choices"]:
            assert set(choice) == {"index", "delta", "finish_reason"}
            assert set(choice["delta"]) <= {"role", "content"}
    assert events[-1]["usage"] == {"total_tokens": 9}


def logprob_provider_transport(seen_requests: list):
    async def body():
        for token in ["Mail ", "bob", "@", "example", ".com"]:
            yield (event_line(logprob_choice(token)) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    def handler(request):
        seen_requests.append(json.loads(request.content))
        if not seen_requests[-1].get("stream"):
            message = {"role": "assistant", "content": "Mail bob@example.com", "refusal": "bob@example.com"}
            logprobs = {"content": [{"token": "bob@example.com", "bytes": [98], "logprob": -1}]}
            choice = {"index": 0, "message": message, "logprobs": logprobs, "finish_reason": "stop"}
            return httpx.Response(200, json={"id": "c1", "choices": [choice], "usage": {"total_tokens": 5}})
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


async def test_logprobs_requested_by_the_client_cannot_reveal_redacted_text():
    seen = []
    request = {**REQUEST, "logprobs": True, "top_logprobs": 2}
    async with gateway_client(logprob_provider_transport(seen)) as client:
        streamed = await client.post("/v1/chat/completions", json=request)
        plain = await client.post("/v1/chat/completions", json={**request, "stream": False})

    assert seen[0]["logprobs"] is True  # the option reached the provider, so it did send logprobs back
    events, done = parse_sse(streamed.content)
    assert done
    assert content_of(events) == "Mail [REDACTED]"
    assert "bob" not in streamed.text
    assert plain.status_code == 200
    assert plain.json() == {
        "id": "c1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Mail [REDACTED]"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 5},
    }


STREAM_ERROR = {"error": {"type": "upstream_error", "message": "LLM provider sent an invalid event"}}


@pytest.mark.parametrize(
    "bad_line",
    [
        "data: null",
        "data: []",
        "data: 42",
        'data: {"choices": null}',
        'data: {"id": "c1"}',
        'data: {"error": {"message": "internal failure for bob@example.com"}}',
        'data: {"choices": [null]}',
        'data: {"choices": [{"index": [0], "delta": {"content": "x"}}]}',
        'data: {"choices": [{"index": 0, "delta": "bob@example.com"}]}',
        'data: {"choices": [{"index": 0, "delta": {"content": ["bob@example.com"]}}]}',
        'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": {"text": "bob@example.com"}}]}',
        "data: not json bob@exam",
    ],
)
async def test_malformed_provider_events_end_the_stream_safely(bad_line):
    lines = [event_line({"index": 0, "delta": {"content": "Hi carol@exam"}}), bad_line, event_line({"index": 0, "delta": {"content": "x"}})]
    raw = await collect(redact_sse_stream(as_async(lines)))
    events, done = parse_sse(raw)

    assert not done
    # Held text is flushed (redacted on its own), then a sanitized error; nothing after the bad event.
    assert content_of(events) == "Hi carol@exam"
    assert events[-1] == STREAM_ERROR
    assert b"bob" not in raw


async def test_malformed_completion_is_a_gateway_error():
    for body in [None, [], {"choices": "x"}, {"choices": [{"message": "bob@example.com"}]}]:
        transport = httpx.MockTransport(lambda request, body=body: httpx.Response(200, json=body))
        async with gateway_client(transport) as client:
            response = await client.post("/v1/chat/completions", json={**REQUEST, "stream": False})
        assert response.status_code == 502
        assert response.json() == {"error": {"type": "upstream_error", "message": "LLM provider returned an invalid response"}}


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, lines: list[str], error: Exception | None = None):
        self.lines, self.error, self.closed = lines, error, False

    async def __aiter__(self):
        for line in self.lines:
            yield (line + "\n").encode()
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize(
    "lines, error",
    [
        (sse_lines(["fine"]), None),
        (["data: null"], None),
        (sse_lines(["half"])[:2], httpx.ReadError("reset")),
    ],
)
async def test_provider_response_is_closed_however_the_stream_ends(lines, error):
    stream = TrackedStream(lines, error)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    async with gateway_client(transport) as client:
        response = await client.post("/v1/chat/completions", json=REQUEST)
    assert response.status_code == 200
    assert stream.closed
