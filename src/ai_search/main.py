"""FastAPI application entry point with lifespan management."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from ai_search.config import settings
from ai_search.models import HealthResponse, SearchResponse, StatsResponse
from ai_search.pipeline.orchestrator import run_pipeline
from ai_search.services.redis_client import RedisClient
from ai_search.services.searxng import SearXNGClient
from ai_search.utils.logging import configure_logging

logger = structlog.get_logger(__name__)

_startup_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load heavy resources at startup and clean up on shutdown."""
    global _startup_time
    configure_logging(settings.environment, settings.log_level)
    log = structlog.get_logger(__name__)

    log.info("ai_search.startup", environment=settings.environment, port=settings.port)

    # Load cross-encoder model once so first request is not cold.
    # Disabled by default on constrained ARM devices where torch wheels may crash
    # with illegal-instruction faults.
    app.state.cross_encoder = None
    if settings.enable_cross_encoder:
        log.info("cross_encoder.loading", model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        try:
            from sentence_transformers import CrossEncoder

            app.state.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            log.info("cross_encoder.loaded")
        except Exception as exc:  # noqa: BLE001
            log.warning("cross_encoder.load_failed", error=str(exc))
            app.state.cross_encoder = None
    else:
        log.info("cross_encoder.disabled")

    # Initialise Redis client.
    app.state.redis = RedisClient(settings.redis_url)
    await app.state.redis.connect()

    _startup_time = time.time()
    log.info("ai_search.ready")

    yield

    log.info("ai_search.shutdown")
    await app.state.redis.disconnect()


app = FastAPI(
    title="AI Search Middleware",
    description="AI-powered search middleware: SearXNG + Claude Haiku + cross-encoder re-ranking.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/search", response_model=SearchResponse, summary="Perform an AI-enhanced search")
async def search(
    q: str = Query(..., min_length=1, max_length=512, description="Search query"),
    limit: int = Query(default=8, ge=1, le=20, description="Maximum results to return"),
    domain_hint: str | None = Query(default=None, description="Optional domain bias hint"),
) -> SearchResponse:
    """Run the full 5-stage search pipeline and return ranked results."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be blank.")

    result = await run_pipeline(
        query=q.strip(),
        limit=limit,
        domain_hint=domain_hint,
        redis=app.state.redis,
        cross_encoder=app.state.cross_encoder,
    )
    return result


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    """Check liveness of Redis, SearXNG, and the cross-encoder model."""
    uptime = time.time() - _startup_time if _startup_time else 0.0

    # Redis
    redis_status = "connected"
    try:
        await app.state.redis.ping()
    except Exception:  # noqa: BLE001
        redis_status = "unavailable"

    # SearXNG
    searxng_status = "reachable"
    try:
        async with SearXNGClient(settings.searxng_url) as client:
            await client.ping()
    except Exception:  # noqa: BLE001
        searxng_status = "unreachable"

    # Cross-encoder
    cross_encoder_status = "loaded" if app.state.cross_encoder is not None else "unavailable"

    overall = (
        "ok"
        if redis_status == "connected" and searxng_status == "reachable"
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        redis=redis_status,
        searxng=searxng_status,
        cross_encoder=cross_encoder_status,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/stats", response_model=StatsResponse, summary="Aggregated query statistics")
async def stats() -> StatsResponse:
    """Return cumulative query statistics from Redis counters."""
    try:
        data = await app.state.redis.get_stats()
    except Exception as exc:  # noqa: BLE001
        logger.warning("stats.redis_error", error=str(exc))
        raise HTTPException(status_code=503, detail="Stats unavailable — Redis error.") from exc

    queries_total: int = data.get("queries_total", 0)
    cache_hits: int = data.get("cache_hits", 0)
    total_latency_ms: float = data.get("total_latency_ms", 0.0)

    cache_hit_rate = cache_hits / queries_total if queries_total else 0.0
    avg_latency_ms = total_latency_ms / queries_total if queries_total else 0.0

    return StatsResponse(
        queries_total=queries_total,
        cache_hit_rate=round(cache_hit_rate, 4),
        avg_latency_ms=round(avg_latency_ms, 1),
        queries_by_intent=data.get("queries_by_intent", {}),
    )


# ---------------------------------------------------------------------------
# Generic error handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    """Catch-all error handler that logs and returns a structured response."""
    logger.error("unhandled_exception", path=str(request.url), error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "error": str(exc)},
    )
