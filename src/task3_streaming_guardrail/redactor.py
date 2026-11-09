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
# full run (not just 13-19 digits) lets sensitive_spans find a card or SSN inside it.
# A longer run is matched in pieces, and a piece must not stop inside an SSN ("1234 123|-45-6789"
# or "123-45|-6789"), or the SSN's tail would be left outside every match. Matching SSNs on their
# own instead would let SSNs and pieces overlap in a chain, so a dash-joined list of SSNs would
# be held back in full while streaming.
NOT_INSIDE_SSN = (
    r"(?!(?<=(?<![0-9])[0-9]{3})-[0-9]{2}-[0-9]{4}(?![0-9]))"
    r"(?!(?<=(?<![0-9])[0-9]{3}-[0-9]{2})-[0-9]{4}(?![0-9]))"
)
DIGIT_RUN = r"(?<![0-9])[0-9](?:[ -]?[0-9]){12,47}(?![0-9])" + NOT_INSIDE_SSN
# Emails and numbers are searched separately and their redactions merged. In one alternation the
# leftmost match wins, so the digit run in "2024 01 15 12345@example.com" would swallow the
# email's local part; searching emails first instead would hide most of "4111 1111 1111 1111@x.com".
EMAIL_PATTERN = re.compile(EMAIL)
NUMBER_PATTERN = re.compile(f"{SSN}|{DIGIT_RUN}")
SSN_PATTERN = re.compile(SSN)

EMAIL_CHARS = frozenset(string.ascii_letters + string.digits + LOCAL_PART_SYMBOLS + "@")
NUMBER_CHARS = frozenset(string.digits + " -")

# Longest possible match: an email of 64 + 1 + 63 + 8 * 64 + 1 + 63 = 704 chars (a digit run is
# at most 95). Holding back at least that much means text already sent can never turn out to be
# the start of a match. A cut never splits a match: an email crossing the window can start 703
# chars before it, and a digit run ending in that email's local part 94 chars before that, so
# `pending` can reach MAX_HOLDBACK + 703 + 94 = 1,501 chars after a feed, per choice.
MAX_HOLDBACK = 704


def luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def find_pii(text: str) -> list[re.Match]:
    """Every email and every SSN/digit-run match. An email can overlap a digit run ending in its local part."""
    return [*EMAIL_PATTERN.finditer(text), *NUMBER_PATTERN.finditer(text)]


def sensitive_spans(match: re.Match) -> list[tuple[int, int]]:
    if match.re is EMAIL_PATTERN:
        return [match.span()]
    # A card or SSN can sit next to unrelated digits, e.g. "4111 1111 1111 1111 12/26" or
    # "555 123 4567 123-45-6789". Redact every SSN and every sequence of whole digit groups
    # that forms a Luhn-valid 13-19 digit card; other digits (order IDs, phones) stay.
    run, offset = match.group(), match.start()
    sensitive = [ssn.span() for ssn in SSN_PATTERN.finditer(run)]
    groups = [group.span() for group in re.finditer(r"[0-9]+", run)]
    for first in range(len(groups)):
        digits = ""
        for start, end in groups[first:]:
            digits += run[start:end]
            if len(digits) > 19:
                break
            if len(digits) >= 13 and luhn_valid(digits):
                sensitive.append((groups[first][0], end))
    return [(offset + start, offset + end) for start, end in sensitive]


def redact_matches(text: str, matches: list[re.Match]) -> str:
    """Replace each stretch covered by overlapping sensitive spans with one [REDACTED]."""
    merged: list[list[int]] = []
    for start, end in sorted(span for match in matches for span in sensitive_spans(match)):
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    output, position = [], 0
    for start, end in merged:
        output += [text[position:start], REDACTED]
        position = end
    output.append(text[position:])
    return "".join(output)


def redact_text(text: str) -> str:
    return redact_matches(text, find_pii(text))


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
        matches = find_pii(self.pending)
        # Never cut through a match; hold the whole value until it's complete. Moving the cut back to
        # an email's start can put it inside a digit run that ends in the local part, so repeat.
        while crossing := [match.start() for match in matches if match.start() < cut < match.end()]:
            cut = min(crossing)
        output = redact_matches(self.pending[:cut], [match for match in matches if match.end() <= cut])
        self.pending = self.pending[cut:]
        return output
