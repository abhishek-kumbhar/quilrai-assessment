"""Incremental PII redaction for streamed text.

Emits text as soon as it can no longer be part of a PII match. Only the trailing run of
characters that could still grow into a match (email characters, or digits/spaces/dashes)
is held back, so latency is roughly one word and memory stays bounded.
"""

import re
import string

REDACTED = "[REDACTED]"

EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
# [0-9] instead of \d so non-ASCII digits don't widen the match.
SSN = r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])"
CARD = r"(?<![0-9])[0-9](?:[ -]?[0-9]){12,18}(?![0-9])"
PII_PATTERN = re.compile(f"(?P<email>{EMAIL})|(?P<ssn>{SSN})|(?P<card>{CARD})")

EMAIL_CHARS = frozenset(string.ascii_letters + string.digits + "._%+-@")
NUMBER_CHARS = frozenset(string.digits + " -")

# Longest valid email is 254 chars; cards with separators are under 40.
MAX_HOLDBACK = 256


def luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def replacement(match: re.Match) -> str:
    # Long digit runs (order IDs, timestamps) are only treated as cards if the Luhn check passes.
    if match.lastgroup == "card":
        digits = re.sub(r"[ -]", "", match.group())
        if not luhn_valid(digits):
            return match.group()
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
