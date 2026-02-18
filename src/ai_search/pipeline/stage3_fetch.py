"""Stage 3 — SearXNG fetch and result deduplication."""

from __future__ import annotations

import difflib
from datetime import date
from typing import Any

import structlog

from ai_search.config import settings
from ai_search.models import EngineConfig, RawResult
from ai_search.services.searxng import SearXNGClient
from ai_search.utils.url_utils import normalize_url

logger = structlog.get_logger(__name__)

_TARGET_UNIQUE = 15
_TITLE_DUP_THRESHOLD = 0.85


def _parse_date(raw: Any) -> date | None:  # noqa: ANN401
    """Attempt to parse a date value from SearXNG into a :class:`date`.

    SearXNG returns dates in many formats (ISO, partial, year-only, etc.).
    Any failure is silently ignored.

    Args:
        raw: Raw date value from SearXNG (str, None, etc.).

    Returns:
        Parsed :class:`date` or ``None``.
    """
    if not raw:
        return None
    if isinstance(raw, date):
        return raw
    from datetime import datetime

    raw_str = str(raw).strip()
    # Map format → expected input length so we slice the string correctly.
    formats: list[tuple[str, int]] = [
        ("%Y-%m-%dT%H:%M:%S%z", 25),
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%d", 10),
        ("%Y-%m", 7),
        ("%Y", 4),
    ]
    for fmt, length in formats:
        try:
            return datetime.strptime(raw_str[:length], fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _is_title_duplicate(a: str, b: str) -> bool:
    """Return True if titles *a* and *b* are near-identical."""
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return ratio > _TITLE_DUP_THRESHOLD


def _deduplicate(results: list[RawResult]) -> list[RawResult]:
    """Remove duplicates by URL and near-identical titles, keeping best-ranked.

    Args:
        results: List of :class:`RawResult` in engine-rank order.

    Returns:
        Deduplicated list capped at :data:`_TARGET_UNIQUE` entries.
    """
    seen_urls: set[str] = set()
    unique: list[RawResult] = []

    for result in results:
        norm_url = normalize_url(result.url)

        # Exact URL duplicate.
        if norm_url in seen_urls:
            continue

        # Near-title duplicate — compare against already-accepted results.
        is_dup = any(_is_title_duplicate(result.title, kept.title) for kept in unique)
        if is_dup:
            continue

        seen_urls.add(norm_url)
        unique.append(result)

        if len(unique) >= _TARGET_UNIQUE:
            break

    return unique


async def fetch_results(
    expanded_query: str,
    engine_config: EngineConfig,
) -> list[RawResult]:
    """Fetch, normalise, and deduplicate results from SearXNG.

    Args:
        expanded_query: The Claude-expanded query string.
        engine_config:  Routing config (engines + categories) from Stage 2.

    Returns:
        Up to :data:`_TARGET_UNIQUE` unique :class:`RawResult` objects.

    Raises:
        httpx.HTTPError: If SearXNG is unreachable (caller should handle).
    """
    async with SearXNGClient(settings.searxng_url) as client:
        raw_items = await client.search(
            query=expanded_query,
            engines=engine_config.engines,
            categories=engine_config.categories,
        )

    results: list[RawResult] = []
    for rank, item in enumerate(raw_items):
        try:
            result = RawResult(
                title=str(item.get("title", "")).strip() or "(no title)",
                url=str(item.get("url", "")),
                content=str(item.get("content", "")),
                engine=str(item.get("engine", "")),
                score=float(item.get("score") or 0.0),
                published_date=_parse_date(item.get("publishedDate")),
                engine_rank=rank,
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch.result_parse_error", rank=rank, error=str(exc))

    logger.info("fetch.raw_results", count=len(results))
    unique = _deduplicate(results)
    logger.info("fetch.unique_results", count=len(unique))
    return unique
