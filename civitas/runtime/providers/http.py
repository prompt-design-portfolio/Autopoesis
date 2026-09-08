"""Shared HTTP machinery for the network providers (Part B §19, §52, §54).

Every network provider needs the same four things — a client, retry with backoff on the errors
that can succeed on a second attempt, a timeout, and credentials that never reach a log. They live
here once so a provider adapter is only the part that is actually vendor-specific.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from civitas.runtime.providers.base import ProviderError, RateLimited

log = logging.getLogger(__name__)

#: Statuses worth retrying: rate limits, and the 5xx family that indicates the request never
#: landed. A 400 is a bug in the request and retrying it just spends money.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

DEFAULT_TIMEOUT_S = 120.0
MAX_RETRIES = 4


class HttpProviderMixin:
    """Request/retry behaviour shared by the network providers."""

    _client: httpx.Client | None = None
    base_url: str = ""
    timeout_s: float = DEFAULT_TIMEOUT_S

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s, follow_redirects=False)
        return self._client

    def _headers(self) -> dict[str, str]:  # pragma: no cover - overridden
        return {}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with bounded retries and jittered exponential backoff.

        Jitter matters under load: without it, every worker that hit the same rate limit retries
        in lockstep and re-creates the burst that caused it.
        """
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        last: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._get_client().post(url, json=payload, headers=self._headers())
            except httpx.TimeoutException:
                last = ProviderError(f"{self.name}: timeout after {self.timeout_s}s",
                                     retryable=True)
            except httpx.HTTPError as exc:
                last = ProviderError(f"{self.name}: transport error: {exc}", retryable=True)
            else:
                if response.status_code < 300:
                    return response.json()
                # The body can echo the request, which can contain a key the caller passed by
                # mistake. Truncate hard and never log it (Part B §40).
                detail = response.text[:500]
                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    last = RateLimited(
                        f"{self.name}: rate limited",
                        float(retry_after) if retry_after and retry_after.isdigit() else None,
                    )
                elif response.status_code in RETRYABLE_STATUS:
                    last = ProviderError(f"{self.name}: {response.status_code}",
                                         retryable=True, status=response.status_code)
                else:
                    raise ProviderError(
                        f"{self.name}: {response.status_code}: {detail}",
                        retryable=False, status=response.status_code,
                    )

            if attempt < MAX_RETRIES - 1:
                delay = min(30.0, 0.5 * (2**attempt)) * (0.5 + random.random())
                explicit = getattr(last, "retry_after_s", None)
                time.sleep(explicit if explicit else delay)

        raise last or ProviderError(f"{self.name}: exhausted retries", retryable=True)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
