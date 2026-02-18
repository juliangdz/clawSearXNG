"""Stage 0 — Redis cache lookup."""

from __future__ import annotations

import hashlib

import structlog

from ai_search.models import SearchResponse
from ai_search.services.redis_client import RedisClient

logger = structlog.get_logger(__name__)


def make_cache_key(query: str) -> str:
    """Return a deterministic SHA256 cache key for *query*.

    Args:
        query: Raw user query string.

    Returns:
        64-character lowercase hex string.
    """
    normalised = query.strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


async def check_cache(query: str, redis: RedisClient) -> SearchResponse | None:
    """Attempt to retrieve a cached :class:`SearchResponse` for *query*.

    Args:
        query: User query string.
        redis: Connected :class:`RedisClient` instance.

    Returns:
        Deserialised :class:`SearchResponse` on cache hit, or ``None``.
    """
    key = make_cache_key(query)
    try:
        data = await redis.get_cache(key)
        if data is not None:
            logger.info("cache.hit", key=key[:8])
            response = SearchResponse.model_validate(data)
            response = response.model_copy(update={"cache_hit": True})
            return response
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache.check_error", error=str(exc))
    return None
