"""Async SearXNG HTTP client."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

# Timeout settings for SearXNG requests (seconds).
_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)


class SearXNGClient:
    """Async HTTP client for a locally running SearXNG instance.

    Designed to be used as an async context manager::

        async with SearXNGClient(base_url) as client:
            results = await client.search(query, engines, categories)

    Args:
        base_url: Base URL of the SearXNG instance, e.g. ``"http://localhost:8888"``.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SearXNGClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=_TIMEOUT)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("SearXNGClient must be used as an async context manager.")
        return self._client

    async def ping(self) -> bool:
        """Perform a lightweight GET / to verify the instance is reachable.

        Returns:
            ``True`` if the server responds with HTTP 200.

        Raises:
            httpx.HTTPError: If the request fails.
        """
        resp = await self._http.get("/", timeout=5.0)
        resp.raise_for_status()
        return True

    async def search(
        self,
        query: str,
        engines: list[str],
        categories: list[str],
    ) -> list[dict[str, Any]]:
        """Fetch search results from SearXNG.

        Args:
            query:      The search query string.
            engines:    List of SearXNG engine identifiers.
            categories: List of SearXNG category names.

        Returns:
            List of raw result dicts from SearXNG (``results`` key).

        Raises:
            httpx.HTTPError: On network or HTTP errors.
        """
        params: dict[str, str] = {
            "q": query,
            "engines": ",".join(engines),
            "categories": ",".join(categories),
            "format": "json",
        }

        logger.debug(
            "searxng.search",
            query=query,
            engines=engines,
            categories=categories,
        )

        resp = await self._http.get("/search", params=params)
        resp.raise_for_status()

        data = resp.json()
        results: list[dict[str, Any]] = data.get("results", [])
        logger.debug("searxng.results_received", count=len(results))
        return results
