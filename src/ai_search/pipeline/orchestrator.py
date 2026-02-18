"""Pipeline orchestrator — wires all 5 stages together."""

from __future__ import annotations

import time
from typing import Any

import structlog

from ai_search.models import SearchResponse
from ai_search.pipeline import stage0_cache, stage1_intel, stage2_router
from ai_search.pipeline import stage3_fetch, stage4a_coarse, stage4b_rerank, stage5_return
from ai_search.services.redis_client import RedisClient

logger = structlog.get_logger(__name__)


async def run_pipeline(
    query: str,
    limit: int,
    domain_hint: str | None,
    redis: RedisClient,
    cross_encoder: Any,  # noqa: ANN401
) -> SearchResponse:
    """Execute the full 5-stage search pipeline for *query*.

    Stages:
        0. Cache lookup (return immediately on hit).
        1. Query intelligence (tiered local rules or Claude Haiku, based on env).
        2. Engine routing by intent.
        3. SearXNG fetch + deduplication.
        4A. Coarse metadata scoring.
        4B. Cross-encoder semantic re-ranking.
        5. Cache write + stats + response construction.

    Args:
        query:        Original user query string.
        limit:        Maximum results to return.
        domain_hint:  Optional domain preference (reserved for future use).
        redis:        Connected Redis client.
        cross_encoder: Loaded CrossEncoder model (may be ``None``).

    Returns:
        :class:`SearchResponse` ready to serialise as JSON.
    """
    t_start = time.perf_counter()

    log = logger.bind(query=query[:80], limit=limit)
    log.info("pipeline.start")

    # ------------------------------------------------------------------ #
    # Stage 0 — Cache lookup                                               #
    # ------------------------------------------------------------------ #
    cached = await stage0_cache.check_cache(query, redis)
    if cached is not None:
        elapsed = (time.perf_counter() - t_start) * 1000
        log.info("pipeline.cache_hit", latency_ms=round(elapsed, 1))
        try:
            await redis.increment_counter("queries_total")
            await redis.increment_counter("cache_hits")
            await redis.increment_intent(cached.intent)
            await redis.increment_counter("total_latency_ms", elapsed)
        except Exception:  # noqa: BLE001
            pass
        return cached

    # ------------------------------------------------------------------ #
    # Stage 1 — Query intelligence                                         #
    # ------------------------------------------------------------------ #
    intelligence = await stage1_intel.analyze_query(query)

    # ------------------------------------------------------------------ #
    # Stage 2 — Engine routing                                             #
    # ------------------------------------------------------------------ #
    engine_config = stage2_router.route(intelligence.intent)

    # ------------------------------------------------------------------ #
    # Stage 3 — Fetch + deduplicate                                        #
    # ------------------------------------------------------------------ #
    try:
        raw_results = await stage3_fetch.fetch_results(
            expanded_query=intelligence.expanded_query,
            engine_config=engine_config,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - t_start) * 1000
        log.error("pipeline.searxng_failed", error=str(exc), latency_ms=round(elapsed, 1))
        from fastapi import HTTPException

        raise HTTPException(
            status_code=502,
            detail=f"SearXNG fetch failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------ #
    # Stage 4A — Coarse filter                                             #
    # ------------------------------------------------------------------ #
    coarse_results = stage4a_coarse.coarse_filter(raw_results)

    # ------------------------------------------------------------------ #
    # Stage 4B — Cross-encoder re-rank                                     #
    # ------------------------------------------------------------------ #
    ranked_results = stage4b_rerank.rerank(
        results=coarse_results,
        expanded_query=intelligence.expanded_query,
        cross_encoder=cross_encoder,
    )

    # ------------------------------------------------------------------ #
    # Stage 5 — Build response + cache                                     #
    # ------------------------------------------------------------------ #
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    response = stage5_return.build_response(
        query=query,
        expanded_query=intelligence.expanded_query,
        intent=intelligence.intent,
        cache_hit=False,
        query_time_ms=elapsed_ms,
        results=ranked_results,
        limit=limit,
    )

    await stage5_return.cache_and_record(
        query=query,
        response=response,
        intent=intelligence.intent,
        query_time_ms=elapsed_ms,
        redis=redis,
    )

    log.info(
        "pipeline.complete",
        intent=intelligence.intent,
        results=len(response.results),
        latency_ms=round(elapsed_ms, 1),
    )
    return response
