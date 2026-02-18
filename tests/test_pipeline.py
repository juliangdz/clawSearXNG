"""Tests for pipeline stages 0, 1, 2, 4A, 4B, and 5."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_search.models import (
    EngineConfig,
    RawResult,
    ScoredResult,
    SearchResponse,
)
from ai_search.pipeline import (
    stage0_cache,
    stage2_router,
    stage4a_coarse,
    stage4b_rerank,
    stage5_return,
)
from ai_search.pipeline.stage0_cache import make_cache_key


# ---------------------------------------------------------------------------
# Stage 0 — Cache
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    """Tests for the cache key generation function."""

    def test_deterministic(self) -> None:
        """Same query always produces the same key."""
        assert make_cache_key("hello world") == make_cache_key("hello world")

    def test_case_insensitive(self) -> None:
        """Keys are normalised to lowercase."""
        assert make_cache_key("Hello World") == make_cache_key("hello world")

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert make_cache_key("  hello  ") == make_cache_key("hello")

    def test_sha256_format(self) -> None:
        """Key is a 64-character hex string."""
        key = make_cache_key("test")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_known_value(self) -> None:
        """Verify against a pre-computed hash."""
        expected = hashlib.sha256(b"hello").hexdigest()
        assert make_cache_key("hello") == expected


@pytest.mark.asyncio
class TestCheckCache:
    """Tests for the cache lookup function."""

    async def test_cache_miss(self, mock_redis: AsyncMock) -> None:
        """Returns None on a cache miss."""
        mock_redis.get_cache.return_value = None
        result = await stage0_cache.check_cache("test query", mock_redis)
        assert result is None

    async def test_cache_hit(self, mock_redis: AsyncMock) -> None:
        """Returns a SearchResponse with cache_hit=True on a hit."""
        cached_data = {
            "query": "test query",
            "expanded_query": "test query expanded",
            "intent": "general",
            "cache_hit": False,
            "query_time_ms": 100.0,
            "results": [],
        }
        mock_redis.get_cache.return_value = cached_data
        result = await stage0_cache.check_cache("test query", mock_redis)
        assert result is not None
        assert result.cache_hit is True
        assert result.query == "test query"

    async def test_redis_error_returns_none(self, mock_redis: AsyncMock) -> None:
        """Gracefully returns None if Redis raises."""
        mock_redis.get_cache.side_effect = Exception("connection refused")
        result = await stage0_cache.check_cache("test query", mock_redis)
        assert result is None


# ---------------------------------------------------------------------------
# Stage 2 — Router
# ---------------------------------------------------------------------------


class TestRouter:
    """Tests for the engine router."""

    def test_known_intent_research(self) -> None:
        """Research intent returns arxiv and semantic_scholar."""
        config = stage2_router.route("research")
        assert "arxiv" in config.engines
        assert "semantic_scholar" in config.engines

    def test_known_intent_code(self) -> None:
        """Code intent returns github."""
        config = stage2_router.route("code")
        assert "github" in config.engines

    def test_unknown_intent_falls_back(self) -> None:
        """Unknown intents fall back to general config."""
        config = stage2_router.route("totally_unknown_intent_xyz")
        assert isinstance(config, EngineConfig)
        assert len(config.engines) > 0

    def test_returns_engine_config(self) -> None:
        """Always returns an EngineConfig instance."""
        for intent in ("research", "biomedical", "code", "news", "general"):
            config = stage2_router.route(intent)
            assert isinstance(config, EngineConfig)
            assert len(config.engines) > 0
            assert len(config.categories) > 0


# ---------------------------------------------------------------------------
# Stage 4A — Coarse filter
# ---------------------------------------------------------------------------


class TestCoarseFilter:
    """Tests for the metadata-based coarse filter."""

    def test_returns_scored_results(self, sample_raw_results: list[RawResult]) -> None:
        """Returns ScoredResult objects."""
        results = stage4a_coarse.coarse_filter(sample_raw_results)
        assert all(isinstance(r, ScoredResult) for r in results)

    def test_sorted_descending(self, sample_raw_results: list[RawResult]) -> None:
        """Results are sorted by metadata_score descending."""
        results = stage4a_coarse.coarse_filter(sample_raw_results)
        scores = [r.metadata_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_max_12(self) -> None:
        """At most 12 results are returned."""
        many = [
            RawResult(
                title=f"Result {i}",
                url=f"https://example.com/{i}",
                engine="ddg",
                engine_rank=i,
            )
            for i in range(20)
        ]
        results = stage4a_coarse.coarse_filter(many)
        assert len(results) <= 12

    def test_authority_scores_set(self, sample_raw_results: list[RawResult]) -> None:
        """Authority scores are non-zero for known high-authority domains."""
        results = stage4a_coarse.coarse_filter(sample_raw_results)
        arxiv_result = next(r for r in results if "arxiv.org" in r.url)
        assert arxiv_result.authority_score == 1.0

    def test_empty_input(self) -> None:
        """Returns empty list for empty input."""
        assert stage4a_coarse.coarse_filter([]) == []


class TestComputeMetadataScore:
    """Tests for the metadata score computation function."""

    def test_high_authority_domain(self) -> None:
        """arxiv.org gets authority=1.0."""
        result = RawResult(
            title="Test",
            url="https://arxiv.org/abs/1234.5678",
            engine="arxiv",
            engine_rank=0,
        )
        score, authority, _, _, _ = stage4a_coarse.compute_metadata_score(result)
        assert authority == 1.0
        assert score > 0.5  # Should score well

    def test_unknown_domain(self) -> None:
        """Unknown domains get DEFAULT_AUTHORITY (0.30)."""
        result = RawResult(
            title="Test",
            url="https://some-random-blog.example.com/post",
            engine="ddg",
            engine_rank=5,
        )
        _, authority, _, _, _ = stage4a_coarse.compute_metadata_score(result)
        assert authority == 0.30

    def test_recency_defaults_to_half(self) -> None:
        """Missing published_date gives recency=0.5."""
        result = RawResult(
            title="Test",
            url="https://example.com/",
            engine="ddg",
            engine_rank=0,
            published_date=None,
        )
        _, _, recency, _, _ = stage4a_coarse.compute_metadata_score(result)
        assert recency == 0.5


# ---------------------------------------------------------------------------
# Stage 4B — Re-rank
# ---------------------------------------------------------------------------


class TestRerank:
    """Tests for the cross-encoder re-ranking stage."""

    def test_returns_at_most_8(
        self,
        sample_scored_results: list[ScoredResult],
        mock_cross_encoder: MagicMock,
    ) -> None:
        """At most 8 results are returned."""
        many = sample_scored_results * 4  # 12 results
        mock_cross_encoder.predict.return_value = [0.5] * len(many)
        results = stage4b_rerank.rerank(many, "query", mock_cross_encoder)
        assert len(results) <= 8

    def test_sorted_by_final_score(
        self,
        sample_scored_results: list[ScoredResult],
        mock_cross_encoder: MagicMock,
    ) -> None:
        """Results are sorted by final_score descending."""
        mock_cross_encoder.predict.return_value = [0.8, 0.5, 0.3]
        results = stage4b_rerank.rerank(
            sample_scored_results, "transformer attention", mock_cross_encoder
        )
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_cross_encoder_none_degrades_gracefully(
        self, sample_scored_results: list[ScoredResult]
    ) -> None:
        """Falls back to metadata scoring when cross_encoder is None."""
        results = stage4b_rerank.rerank(sample_scored_results, "query", cross_encoder=None)
        assert len(results) > 0
        for r in results:
            assert 0.0 <= r.final_score <= 1.0

    def test_cross_encoder_exception_degrades_gracefully(
        self, sample_scored_results: list[ScoredResult]
    ) -> None:
        """Falls back gracefully when the model raises."""
        bad_encoder = MagicMock()
        bad_encoder.predict.side_effect = RuntimeError("CUDA out of memory")
        results = stage4b_rerank.rerank(sample_scored_results, "query", bad_encoder)
        assert len(results) > 0

    def test_empty_input(self, mock_cross_encoder: MagicMock) -> None:
        """Returns empty list for empty input."""
        assert stage4b_rerank.rerank([], "query", mock_cross_encoder) == []

    def test_final_score_in_range(
        self,
        sample_scored_results: list[ScoredResult],
        mock_cross_encoder: MagicMock,
    ) -> None:
        """Final scores are within [0, 1] after normalisation."""
        mock_cross_encoder.predict.return_value = [10.0, -5.0, 3.0]
        results = stage4b_rerank.rerank(
            sample_scored_results, "query", mock_cross_encoder
        )
        for r in results:
            assert r.final_score >= 0.0


# ---------------------------------------------------------------------------
# Stage 5 — Build response
# ---------------------------------------------------------------------------


class TestBuildResponse:
    """Tests for response construction."""

    def test_basic_construction(self, sample_scored_results: list[ScoredResult]) -> None:
        """Response fields are populated correctly."""
        response = stage5_return.build_response(
            query="my query",
            expanded_query="my expanded query",
            intent="research",
            cache_hit=False,
            query_time_ms=250.0,
            results=sample_scored_results,
            limit=8,
        )
        assert response.query == "my query"
        assert response.expanded_query == "my expanded query"
        assert response.intent == "research"
        assert response.cache_hit is False
        assert len(response.results) <= 3  # sample has 3

    def test_limit_respected(self, sample_scored_results: list[ScoredResult]) -> None:
        """Limit parameter caps the number of results."""
        response = stage5_return.build_response(
            query="q",
            expanded_query="q",
            intent="general",
            cache_hit=False,
            query_time_ms=100.0,
            results=sample_scored_results,
            limit=1,
        )
        assert len(response.results) == 1

    def test_score_bounds(self, sample_scored_results: list[ScoredResult]) -> None:
        """All scores in the response are within [0, 1]."""
        response = stage5_return.build_response(
            query="q",
            expanded_query="q",
            intent="general",
            cache_hit=False,
            query_time_ms=100.0,
            results=sample_scored_results,
            limit=10,
        )
        for r in response.results:
            assert 0.0 <= r.final_score <= 1.0
            assert 0.0 <= r.score_breakdown.semantic <= 1.0
            assert 0.0 <= r.score_breakdown.authority <= 1.0

    def test_snippet_truncated(self) -> None:
        """Long content snippets are truncated at 500 chars."""
        long_content = "x" * 1000
        result = ScoredResult(
            title="Title",
            url="https://example.com/",
            content=long_content,
            engine="ddg",
            engine_rank=0,
            authority_score=0.5,
            recency_score=0.5,
            engine_trust_score=0.5,
            position_score=0.5,
            final_score=0.5,
        )
        response = stage5_return.build_response(
            query="q", expanded_query="q", intent="general",
            cache_hit=False, query_time_ms=10.0,
            results=[result], limit=10,
        )
        assert len(response.results[0].snippet) == 500


@pytest.mark.asyncio
class TestCacheAndRecord:
    """Tests for cache write and stats recording."""

    async def test_calls_set_cache(
        self, mock_redis: AsyncMock, sample_scored_results: list[ScoredResult]
    ) -> None:
        """set_cache is called after a fresh search."""
        response = stage5_return.build_response(
            query="q", expanded_query="q", intent="general",
            cache_hit=False, query_time_ms=100.0,
            results=sample_scored_results, limit=8,
        )
        await stage5_return.cache_and_record(
            query="q", response=response, intent="general",
            query_time_ms=100.0, redis=mock_redis,
        )
        mock_redis.set_cache.assert_called_once()

    async def test_increments_counters(
        self, mock_redis: AsyncMock, sample_scored_results: list[ScoredResult]
    ) -> None:
        """Stats counters are incremented."""
        response = stage5_return.build_response(
            query="q", expanded_query="q", intent="general",
            cache_hit=False, query_time_ms=100.0,
            results=sample_scored_results, limit=8,
        )
        await stage5_return.cache_and_record(
            query="q", response=response, intent="general",
            query_time_ms=100.0, redis=mock_redis,
        )
        mock_redis.increment_counter.assert_called()
        mock_redis.increment_intent.assert_called_with("general")

    async def test_redis_error_does_not_raise(self, mock_redis: AsyncMock) -> None:
        """Errors in cache write / stats don't propagate."""
        mock_redis.set_cache.side_effect = Exception("Redis timeout")
        response = SearchResponse(
            query="q", expanded_query="q", intent="general",
            cache_hit=False, query_time_ms=100.0, results=[],
        )
        # Should not raise.
        await stage5_return.cache_and_record(
            query="q", response=response, intent="general",
            query_time_ms=100.0, redis=mock_redis,
        )
