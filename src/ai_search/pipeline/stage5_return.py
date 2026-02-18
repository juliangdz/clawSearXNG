"""Stage 5 — Cache write, stats tracking, and response construction."""

from __future__ import annotations

import structlog

from ai_search.config import settings
from ai_search.models import (
    ScoreBreakdown,
    SearchResponse,
    SearchResult,
    ScoredResult,
)
from ai_search.pipeline.stage0_cache import make_cache_key
from ai_search.services.redis_client import RedisClient
from ai_search.utils.url_utils import extract_domain

logger = structlog.get_logger(__name__)


def _format_date(result: ScoredResult) -> str | None:
    """Format the published_date to a display string (YYYY-MM)."""
    if result.published_date is None:
        return None
    try:
        return result.published_date.strftime("%Y-%m")
    except Exception:  # noqa: BLE001
        return None


def build_response(
    query: str,
    expanded_query: str,
    intent: str,
    cache_hit: bool,
    query_time_ms: float,
    results: list[ScoredResult],
    limit: int,
) -> SearchResponse:
    """Construct the public :class:`SearchResponse` from scored pipeline results.

    Args:
        query:          Original user query.
        expanded_query: Claude-expanded query.
        intent:         Detected query intent.
        cache_hit:      Whether the response came from cache.
        query_time_ms:  Total wall-clock time for the pipeline.
        results:        Scored and ranked results.
        limit:          Maximum number of results to include.

    Returns:
        Ready-to-serialise :class:`SearchResponse`.
    """
    search_results: list[SearchResult] = []
    for r in results[:limit]:
        search_results.append(
            SearchResult(
                title=r.title,
                url=r.url,
                snippet=r.content[:500] if r.content else "",
                domain=extract_domain(r.url),
                source_engine=r.engine,
                published_date=_format_date(r),
                final_score=round(min(max(r.final_score, 0.0), 1.0), 4),
                score_breakdown=ScoreBreakdown(
                    semantic=round(min(max(r.semantic_score, 0.0), 1.0), 4),
                    authority=round(min(max(r.authority_score, 0.0), 1.0), 4),
                    recency=round(min(max(r.recency_score, 0.0), 1.0), 4),
                    engine_trust=round(min(max(r.engine_trust_score, 0.0), 1.0), 4),
                    position=round(min(max(r.position_score, 0.0), 1.0), 4),
                ),
            )
        )

    return SearchResponse(
        query=query,
        expanded_query=expanded_query,
        intent=intent,
        cache_hit=cache_hit,
        query_time_ms=round(query_time_ms, 1),
        results=search_results,
    )


async def cache_and_record(
    query: str,
    response: SearchResponse,
    intent: str,
    query_time_ms: float,
    redis: RedisClient,
) -> None:
    """Write result to cache and update stats counters.

    Args:
        query:         Original user query.
        response:      Completed search response.
        intent:        Detected intent (for per-intent counter).
        query_time_ms: Pipeline latency in milliseconds.
        redis:         Connected :class:`RedisClient` instance.
    """
    # Cache the result.
    key = make_cache_key(query)
    ttl = settings.cache_ttl_hours * 3600
    try:
        await redis.set_cache(key, response.model_dump(mode="json"), ttl)
        logger.debug("cache.stored", key=key[:8], ttl_hours=settings.cache_ttl_hours)
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache.store_failed", error=str(exc))

    # Update stats counters.
    try:
        await redis.increment_counter("queries_total")
        await redis.increment_intent(intent)
        await redis.increment_counter("total_latency_ms", query_time_ms)
    except Exception as exc:  # noqa: BLE001
        logger.warning("stats.increment_failed", error=str(exc))
