"""Incremental PII redaction for streamed text.

Emits text as soon as it can no longer be part of a PII match. Only the trailing run of
characters that could still grow into a match (email characters, or digits/spaces/dashes)
is held back, so latency is roughly one word and memory stays bounded (see MAX_HOLDBACK).
"""

import re
import string

REDACTED = "[REDACTED]"

# Local part: the RFC 5322 atom characters, starting with a letter or digit so leading Markdown
# (`**`, a backtick, a quote) stays outside the match. Length limits (local part 64, labels 63)
# keep matching linear on long runs and bound how long a single match can get.
LOCAL_PART_SYMBOLS = ".!#$%&'*+/=?^_`{|}~-"
LOCAL_PART = f"[A-Za-z0-9][A-Za-z0-9{re.escape(LOCAL_PART_SYMBOLS)}]{{0,63}}"
EMAIL = LOCAL_PART + r"@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,63}"
# [0-9] instead of \d so non-ASCII digits don't widen the match.
SSN = r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])"
# A whole run of 13-48 digits with optional single space/dash separators. Matching the
# full run (not just 13-19 digits) lets redact_digit_run find a card or SSN inside it.
DIGIT_RUN = r"(?<![0-9])[0-9](?:[ -]?[0-9]){12,47}(?![0-9])"
PII_PATTERN = re.compile(f"(?P<email>{EMAIL})|(?P<ssn>{SSN})|(?P<digits>{DIGIT_RUN})")
SSN_PATTERN = re.compile(SSN)

EMAIL_CHARS = frozenset(string.ascii_letters + string.digits + LOCAL_PART_SYMBOLS + "@")
NUMBER_CHARS = frozenset(string.digits + " -")

# Longest possible match: an email of 64 + 1 + 63 + 8 * 64 + 1 + 63 = 704 chars (a digit run is
# at most 95). Holding back at least that much means text already sent can never turn out to be
# the start of a match. Because a cut never splits a match, `pending` can reach
# MAX_HOLDBACK + 703 = 1,407 chars after a feed, per choice.
MAX_HOLDBACK = 704


def luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def redact_digit_run(run: str) -> str:
    # A card or SSN can sit next to unrelated digits, e.g. "4111 1111 1111 1111 12/26" or
    # "555 123 4567 123-45-6789". Redact every SSN and every sequence of whole digit groups
    # that forms a Luhn-valid 13-19 digit card; other digits (order IDs, phones) stay.
    sensitive = [match.span() for match in SSN_PATTERN.finditer(run)]
    groups = [match.span() for match in re.finditer(r"[0-9]+", run)]
    for first in range(len(groups)):
        digits = ""
        for start, end in groups[first:]:
            digits += run[start:end]
            if len(digits) > 19:
                break
            if len(digits) >= 13 and luhn_valid(digits):
                sensitive.append((groups[first][0], end))

    masked = [False] * len(run)
    for start, end in sensitive:
        masked[start:end] = [True] * (end - start)

    output = []
    for index, char in enumerate(run):
        if not masked[index]:
            output.append(char)
        elif index == 0 or not masked[index - 1]:
            output.append(REDACTED)
    return "".join(output)


def replacement(match: re.Match) -> str:
    if match.lastgroup == "digits":
        return redact_digit_run(match.group())
    return REDACTED


def redact_text(text: str) -> str:
    return PII_PATTERN.sub(replacement, text)


def trailing_run_start(text: str, allowed: frozenset[str]) -> int:
    index = len(text)
    while index > 0 and text[index - 1] in allowed:
        index -= 1
    return index


class StreamRedactor:
    def __init__(self) -> None:
        self.pending = ""

    def feed(self, chunk: str) -> str:
        """Add a chunk and return whatever text is now safe to emit (already redacted)."""
        self.pending += chunk
        cut = min(trailing_run_start(self.pending, EMAIL_CHARS), trailing_run_start(self.pending, NUMBER_CHARS))
        cut = max(cut, len(self.pending) - MAX_HOLDBACK)
        return self._emit_until(cut)

    def flush(self) -> str:
        """Call at end of stream to emit the held tail."""
        return self._emit_until(len(self.pending))

    def _emit_until(self, cut: int) -> str:
        output = []
        position = 0
        for match in PII_PATTERN.finditer(self.pending):
            if match.end() > cut:
                # Never cut through a match; hold the whole value until it's complete.
                cut = min(cut, match.start())
                break
            output.append(self.pending[position : match.start()])
            output.append(replacement(match))
            position = match.end()

        output.append(self.pending[position:cut])
        self.pending = self.pending[cut:]
        return "".join(output)
