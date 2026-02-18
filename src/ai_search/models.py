"""Pydantic v2 data models for the AI Search pipeline."""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Intermediate pipeline models
# ---------------------------------------------------------------------------


class RawResult(BaseModel):
    """A single result as returned by SearXNG before any scoring."""

    title: str
    url: str
    content: str = ""
    engine: str = ""
    score: float = 0.0
    published_date: Optional[date] = None
    engine_rank: int = 0


class ScoredResult(RawResult):
    """RawResult enriched with per-component scores."""

    metadata_score: float = 0.0
    semantic_score: float = 0.0
    authority_score: float = 0.0
    recency_score: float = 0.0
    engine_trust_score: float = 0.0
    position_score: float = 0.0
    final_score: float = 0.0


class QueryIntelligence(BaseModel):
    """Output from the Claude Haiku query analysis stage."""

    intent: str = "general"
    expanded_query: str
    rewritten_query: str


class EngineConfig(BaseModel):
    """Engine routing config for a given intent."""

    engines: list[str]
    categories: list[str]


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    """Per-result score breakdown exposed in the API response."""

    semantic: float = Field(..., ge=0.0, le=1.0)
    authority: float = Field(..., ge=0.0, le=1.0)
    recency: float = Field(..., ge=0.0, le=1.0)
    engine_trust: float = Field(..., ge=0.0, le=1.0)
    position: float = Field(..., ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """A single result in the search response."""

    title: str
    url: str
    snippet: str
    domain: str
    source_engine: str
    published_date: Optional[str] = None
    final_score: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: ScoreBreakdown


class SearchResponse(BaseModel):
    """Top-level response from GET /search."""

    query: str
    expanded_query: str
    intent: str
    cache_hit: bool
    query_time_ms: float
    results: list[SearchResult]


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str
    redis: str
    searxng: str
    cross_encoder: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Response from GET /stats."""

    queries_total: int
    cache_hit_rate: float
    avg_latency_ms: float
    queries_by_intent: dict[str, int]
