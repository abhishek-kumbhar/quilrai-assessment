"""Primary/secondary model routing with a hard deadline on the primary call."""

import asyncio
import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("task4_model_router")

PRIMARY_TIMEOUT_SECONDS = 3.0
SECONDARY_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    model: str
    api_key: str = ""


class UpstreamError(Exception):
    """Internal error describing why a provider call failed. Never sent to clients as-is."""

    def __init__(self, provider: str, reason: str, retryable: bool = True):
        super().__init__(f"{provider}: {reason}")
        self.provider = provider
        self.reason = reason
        self.retryable = retryable


class ModelRouter:
    def __init__(
        self,
        primary: Provider,
        secondary: Provider,
        client: httpx.AsyncClient,
        primary_timeout: float = PRIMARY_TIMEOUT_SECONDS,
        secondary_timeout: float = SECONDARY_TIMEOUT_SECONDS,
    ):
        self.primary = primary
        self.secondary = secondary
        self.client = client
        self.primary_timeout = primary_timeout
        self.secondary_timeout = secondary_timeout

    async def complete(self, payload: dict) -> tuple[dict, str]:
        """Return (completion, provider name). Falls back once on 429, timeout, 5xx or network errors."""
        try:
            return await self._call(self.primary, payload, self.primary_timeout), self.primary.name
        except UpstreamError as exc:
            if not exc.retryable:
                raise
            logger.warning("primary provider failed (%s), falling back to %s", exc.reason, self.secondary.name)

        return await self._call(self.secondary, payload, self.secondary_timeout), self.secondary.name

    async def _call(self, provider: Provider, payload: dict, timeout: float) -> dict:
        headers = {"authorization": f"Bearer {provider.api_key}"} if provider.api_key else {}
        try:
            # asyncio.timeout bounds the whole exchange (connect + headers + body). httpx's own
            # timeouts are per phase, so a slowly trickling response could otherwise exceed it.
            # On expiry the in-flight request is cancelled, so a late primary response can't
            # race with the fallback call.
            async with asyncio.timeout(timeout):
                response = await self.client.post(provider.url, json={**payload, "model": provider.model}, headers=headers)
        except TimeoutError:
            raise UpstreamError(provider.name, f"timed out after {timeout:.1f}s") from None
        except httpx.TimeoutException:
            raise UpstreamError(provider.name, "http timeout") from None
        except httpx.HTTPError as exc:
            raise UpstreamError(provider.name, f"network error: {exc!r}") from None

        if response.status_code == 429:
            raise UpstreamError(provider.name, "rate limited (429)")
        if response.status_code >= 500:
            raise UpstreamError(provider.name, f"server error ({response.status_code})")
        if response.status_code != 200:
            # Other 4xx means the request itself was refused; the backup would likely refuse it too.
            raise UpstreamError(provider.name, f"rejected ({response.status_code})", retryable=False)

        try:
            completion = response.json()
        except ValueError:
            raise UpstreamError(provider.name, "invalid JSON response") from None
        if not isinstance(completion, dict) or not completion.get("choices"):
            raise UpstreamError(provider.name, "response missing choices")
        return completion
