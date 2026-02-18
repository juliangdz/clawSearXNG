"""Stage 4B — Cross-encoder semantic re-ranking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ai_search.models import ScoredResult

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

_TOP_N = 8


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1].

    Args:
        scores: Raw cross-encoder logit scores.

    Returns:
        Normalised floats. Returns all zeros for constant inputs.
    """
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span == 0:
        return [0.5] * len(scores)
    return [(s - lo) / span for s in scores]


def rerank(
    results: list[ScoredResult],
    expanded_query: str,
    cross_encoder: Any,  # noqa: ANN401
) -> list[ScoredResult]:
    """Re-rank *results* using the cross-encoder model and return the top N.

    Falls back to metadata-score ordering if the cross-encoder is
    unavailable or raises.

    Args:
        results:        Coarse-filtered results from Stage 4A.
        expanded_query: Expanded query string from Stage 1.
        cross_encoder:  Loaded ``CrossEncoder`` instance (or ``None``).

    Returns:
        Up to :data:`_TOP_N` :class:`ScoredResult` with ``final_score`` set.
    """
    if not results:
        return []

    # Attempt semantic scoring.
    semantic_scores: list[float] = [0.5] * len(results)

    if cross_encoder is not None:
        try:
            pairs = [(expanded_query, r.title) for r in results]
            raw_scores: list[float] = list(cross_encoder.predict(pairs))
            semantic_scores = _normalize(raw_scores)
            logger.debug("rerank.semantic_scores_computed", count=len(semantic_scores))
        except Exception as exc:  # noqa: BLE001
            logger.warning("rerank.cross_encoder_failed", error=str(exc))
            # Fall back to uniform semantic scores.

    # Compute final weighted scores.
    for result, sem in zip(results, semantic_scores):
        result.semantic_score = sem
        result.final_score = (
            0.45 * sem
            + 0.20 * result.authority_score
            + 0.15 * result.recency_score
            + 0.10 * result.engine_trust_score
            + 0.10 * result.position_score
        )

    results.sort(key=lambda r: r.final_score, reverse=True)
    top = results[:_TOP_N]
    logger.info("rerank.done", kept=len(top))
    return top
