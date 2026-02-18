"""Tests for Stage 4B cross-encoder re-ranking and score normalisation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_search.models import ScoredResult
from ai_search.pipeline.stage4b_rerank import _normalize, rerank


# ---------------------------------------------------------------------------
# Score normalisation
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for the min-max normalisation helper."""

    def test_empty_list(self) -> None:
        assert _normalize([]) == []

    def test_all_same_values(self) -> None:
        """Constant input maps to 0.5 (avoid division by zero)."""
        result = _normalize([3.0, 3.0, 3.0])
        assert result == [0.5, 0.5, 0.5]

    def test_min_maps_to_zero(self) -> None:
        result = _normalize([0.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0)

    def test_max_maps_to_one(self) -> None:
        result = _normalize([0.0, 5.0, 10.0])
        assert result[-1] == pytest.approx(1.0)

    def test_middle_value(self) -> None:
        result = _normalize([0.0, 5.0, 10.0])
        assert result[1] == pytest.approx(0.5)

    def test_negative_scores(self) -> None:
        """Works correctly with negative logits from cross-encoder."""
        result = _normalize([-10.0, 0.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_single_element(self) -> None:
        """Single element maps to 0.5."""
        result = _normalize([42.0])
        assert result == [0.5]


# ---------------------------------------------------------------------------
# Rerank function
# ---------------------------------------------------------------------------


def _make_scored_result(idx: int, score: float = 0.5) -> ScoredResult:
    """Helper to create a ScoredResult for testing."""
    return ScoredResult(
        title=f"Result {idx}",
        url=f"https://example.com/{idx}",
        content=f"Content for result {idx}.",
        engine="ddg",
        engine_rank=idx,
        metadata_score=score,
        authority_score=0.5,
        recency_score=0.5,
        engine_trust_score=0.5,
        position_score=0.5,
    )


class TestRerank:
    """Tests for the re-ranking stage."""

    def test_empty_returns_empty(self) -> None:
        """Empty input returns empty output."""
        encoder = MagicMock()
        assert rerank([], "query", encoder) == []

    def test_top_8_limit(self) -> None:
        """At most 8 results returned."""
        encoder = MagicMock()
        encoder.predict.return_value = [float(i) for i in range(12)]
        results = [_make_scored_result(i) for i in range(12)]
        out = rerank(results, "query", encoder)
        assert len(out) <= 8

    def test_final_score_computed(self) -> None:
        """final_score is set on all returned results."""
        encoder = MagicMock()
        encoder.predict.return_value = [0.9, 0.5, 0.1]
        results = [_make_scored_result(i) for i in range(3)]
        out = rerank(results, "query", encoder)
        for r in out:
            assert r.final_score > 0.0

    def test_sorted_descending(self) -> None:
        """Results are returned in descending final_score order."""
        encoder = MagicMock()
        encoder.predict.return_value = [3.0, 1.0, 5.0]
        results = [_make_scored_result(i) for i in range(3)]
        out = rerank(results, "query", encoder)
        scores = [r.final_score for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_semantic_score_set(self) -> None:
        """semantic_score field is set after re-ranking."""
        encoder = MagicMock()
        encoder.predict.return_value = [0.8, 0.4]
        results = [_make_scored_result(i) for i in range(2)]
        out = rerank(results, "query", encoder)
        for r in out:
            assert r.semantic_score >= 0.0

    def test_none_encoder_uses_uniform_semantic(self) -> None:
        """When cross_encoder is None, semantic score defaults to 0.5."""
        results = [_make_scored_result(i) for i in range(3)]
        out = rerank(results, "query", cross_encoder=None)
        for r in out:
            assert r.semantic_score == pytest.approx(0.5)

    def test_encoder_exception_uses_uniform_semantic(self) -> None:
        """When the encoder raises, falls back to uniform scores."""
        encoder = MagicMock()
        encoder.predict.side_effect = RuntimeError("model error")
        results = [_make_scored_result(i) for i in range(3)]
        out = rerank(results, "query", encoder)
        assert len(out) > 0
        for r in out:
            assert r.semantic_score == pytest.approx(0.5)

    def test_final_score_weights(self) -> None:
        """Final score follows the specified formula weights."""
        encoder = MagicMock()
        # Only one result — semantic normalises to 0.5 (uniform), check math.
        results = [
            ScoredResult(
                title="T",
                url="https://arxiv.org/abs/1",
                engine="arxiv",
                engine_rank=0,
                authority_score=1.0,
                recency_score=1.0,
                engine_trust_score=1.0,
                position_score=1.0,
            )
        ]
        encoder.predict.return_value = [1.0]  # single result → normalises to 0.5
        out = rerank(results, "query", encoder)
        r = out[0]
        expected = 0.45 * 0.5 + 0.20 * 1.0 + 0.15 * 1.0 + 0.10 * 1.0 + 0.10 * 1.0
        assert r.final_score == pytest.approx(expected, abs=1e-4)
