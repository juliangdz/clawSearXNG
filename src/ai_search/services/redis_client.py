"""Async Redis client wrapper."""

from __future__ import annotations

import json
from typing import Any

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


class RedisClient:
    """Thin wrapper around ``redis.asyncio.Redis`` with typed helpers.

    Args:
        url: Redis connection URL, e.g. ``"redis://localhost:6379/0"``.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._redis: Redis | None = None

    async def connect(self) -> None:
        """Open the connection pool."""
        self._redis = Redis.from_url(self._url, decode_responses=True)
        logger.info("redis.connected", url=self._url)

    async def disconnect(self) -> None:
        """Close the connection pool gracefully."""
        if self._redis:
            await self._redis.aclose()
            logger.info("redis.disconnected")

    @property
    def _r(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("RedisClient not connected — call connect() first.")
        return self._redis

    async def ping(self) -> bool:
        """Return True if Redis responds to PING."""
        return await self._r.ping()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def get_cache(self, key: str) -> dict[str, Any] | None:
        """Fetch a JSON-serialised cached search result.

        Args:
            key: Cache key (SHA256 hex string).

        Returns:
            Deserialised dict or ``None`` on miss / error.
        """
        try:
            raw = await self._r.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis.cache_get_error", key=key, error=str(exc))
            return None

    async def set_cache(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        """Store a JSON-serialisable value with a TTL.

        Args:
            key:         Cache key.
            value:       Dict to serialise.
            ttl_seconds: Expiry in seconds.
        """
        try:
            await self._r.setex(key, ttl_seconds, json.dumps(value))
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis.cache_set_error", key=key, error=str(exc))

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    async def increment_counter(self, name: str, amount: float = 1.0) -> None:
        """Atomically increment a named counter (stored as a float-compatible string).

        Args:
            name:   Redis key for the counter.
            amount: Increment amount (default 1).
        """
        try:
            await self._r.incrbyfloat(name, amount)
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis.increment_error", name=name, error=str(exc))

    async def increment_intent(self, intent: str) -> None:
        """Increment the per-intent query counter.

        Args:
            intent: Intent label, e.g. ``"research"``.
        """
        await self.increment_counter(f"queries_by_intent:{intent}")

    async def get_stats(self) -> dict[str, Any]:
        """Collect all stats counters from Redis.

        Returns:
            Dict with keys ``queries_total``, ``cache_hits``,
            ``total_latency_ms``, and ``queries_by_intent``.
        """
        queries_total = float(await self._r.get("queries_total") or 0)
        cache_hits = float(await self._r.get("cache_hits") or 0)
        total_latency = float(await self._r.get("total_latency_ms") or 0)

        # Collect per-intent counters via SCAN.
        intents: dict[str, int] = {}
        async for key in self._r.scan_iter("queries_by_intent:*"):
            intent_name = key.split(":", 1)[1]
            val = await self._r.get(key)
            intents[intent_name] = int(float(val or 0))

        return {
            "queries_total": int(queries_total),
            "cache_hits": int(cache_hits),
            "total_latency_ms": total_latency,
            "queries_by_intent": intents,
        }
