"""Shared pytest fixtures and configuration."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_search.models import EngineConfig, QueryIntelligence, RawResult, ScoredResult


# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------

pytest_plugins = ("pytest_asyncio",)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_result() -> RawResult:
    """A single well-formed raw result."""
    return RawResult(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        content="The dominant sequence transduction models are based on complex RNNs.",
        engine="arxiv",
        score=0.9,
        published_date=date(2017, 6, 12),
        engine_rank=0,
    )


@pytest.fixture
def sample_raw_results(sample_raw_result: RawResult) -> list[RawResult]:
    """Small list of raw results for pipeline testing."""
    return [
        sample_raw_result,
        RawResult(
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            url="https://arxiv.org/abs/1810.04805",
            content="We introduce BERT, a new language representation model.",
            engine="arxiv",
            score=0.85,
            published_date=date(2018, 10, 11),
            engine_rank=1,
        ),
        RawResult(
            title="GPT-4 Technical Report",
            url="https://openai.com/research/gpt-4",
            content="GPT-4 is a large multimodal model.",
            engine="ddg",
            score=0.7,
            published_date=date(2023, 3, 15),
            engine_rank=2,
        ),
    ]


@pytest.fixture
def sample_scored_results(sample_raw_results: list[RawResult]) -> list[ScoredResult]:
    """Pre-scored results for re-ranking tests."""
    return [
        ScoredResult(
            **r.model_dump(),
            metadata_score=0.8 - i * 0.1,
            authority_score=0.9,
            recency_score=0.7,
            engine_trust_score=1.0,
            position_score=0.8,
        )
        for i, r in enumerate(sample_raw_results)
    ]


@pytest.fixture
def sample_intelligence() -> QueryIntelligence:
    """Query intelligence for a research-type query."""
    return QueryIntelligence(
        intent="research",
        expanded_query="transformer attention mechanism neural network 2017 2018",
        rewritten_query="transformer attention mechanism",
    )


@pytest.fixture
def sample_engine_config() -> EngineConfig:
    """Engine config for the research intent."""
    return EngineConfig(engines=["arxiv", "semantic_scholar", "ddg"], categories=["science"])


# ---------------------------------------------------------------------------
# Mock service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client that doesn't require a real Redis instance."""
    mock = AsyncMock()
    mock.get_cache.return_value = None  # cache miss by default
    mock.set_cache.return_value = None
    mock.ping.return_value = True
    mock.increment_counter.return_value = None
    mock.increment_intent.return_value = None
    mock.get_stats.return_value = {
        "queries_total": 10,
        "cache_hits": 3,
        "total_latency_ms": 8200.0,
        "queries_by_intent": {"research": 7, "general": 3},
    }
    return mock


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    """Mock CrossEncoder that returns deterministic scores."""
    mock = MagicMock()
    mock.predict.return_value = [0.8, 0.6, 0.4]
    return mock
