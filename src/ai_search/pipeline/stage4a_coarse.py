"""Stage 4A — Coarse metadata-based scoring and filtering."""

from __future__ import annotations

from datetime import date
from math import log

import structlog

from ai_search.models import RawResult, ScoredResult
from ai_search.utils.domain_weights import (
    HEALTHCARE_ML_DOMAINS,
    get_authority,
    get_engine_trust,
)
from ai_search.utils.url_utils import extract_domain

logger = structlog.get_logger(__name__)

_TOP_N = 12


def _recency_score(published_date: date | None) -> float:
    """Convert a publication date to a [0, 1] recency score.

    Args:
        published_date: Date of publication or ``None`` if unknown.

    Returns:
        1.0 for today, decaying towards 0 for older dates.
        Defaults to 0.5 when unknown.
    """
    if published_date is None:
        return 0.5
    try:
        days_old = (date.today() - published_date).days
        return 1.0 / (1.0 + days_old / 365.0)
    except Exception:  # noqa: BLE001
        return 0.5


def _position_score(engine_rank: int) -> float:
    """Convert a 0-based engine rank to a [0, 1] position score.

    Args:
        engine_rank: 0-based rank within the engine's result list.

    Returns:
        Score using logarithmic dampening so top ranks score higher.
    """
    return 1.0 / log(1.0 + engine_rank + 1.0)


def compute_metadata_score(result: RawResult) -> tuple[float, float, float, float, float]:
    """Compute component metadata scores for *result*.

    Args:
        result: Raw result to score.

    Returns:
        Tuple of ``(metadata_score, authority, recency, engine_trust, position)``.
    """
    domain = extract_domain(result.url)
    authority = get_authority(domain)
    recency = _recency_score(result.published_date)
    engine_trust = get_engine_trust(result.engine)
    position = _position_score(result.engine_rank)
    domain_boost = 0.15 if domain in HEALTHCARE_ML_DOMAINS else 0.0

    score = (
        0.35 * authority
        + 0.20 * recency
        + 0.25 * engine_trust
        + 0.20 * position
        + domain_boost
    )

    return score, authority, recency, engine_trust, position


def coarse_filter(results: list[RawResult]) -> list[ScoredResult]:
    """Score all *results* by metadata and return the top :data:`_TOP_N`.

    Args:
        results: Deduplicated raw results from Stage 3.

    Returns:
        Up to :data:`_TOP_N` :class:`ScoredResult` objects, sorted by
        descending metadata score.
    """
    scored: list[ScoredResult] = []
    for r in results:
        meta_score, authority, recency, engine_trust, position = compute_metadata_score(r)
        scored.append(
            ScoredResult(
                **r.model_dump(),
                metadata_score=meta_score,
                authority_score=authority,
                recency_score=recency,
                engine_trust_score=engine_trust,
                position_score=position,
            )
        )

    scored.sort(key=lambda x: x.metadata_score, reverse=True)
    top = scored[:_TOP_N]
    logger.info("coarse.filtered", kept=len(top), total=len(scored))
    return top
