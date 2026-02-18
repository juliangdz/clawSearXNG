"""Stage 1 — Claude Haiku query intelligence."""

from __future__ import annotations

import structlog

from ai_search.models import QueryIntelligence
from ai_search.services.claude import ClaudeClient

logger = structlog.get_logger(__name__)

# Module-level singleton (created lazily so tests can patch it easily).
_claude_client: ClaudeClient | None = None


def _get_client() -> ClaudeClient:
    global _claude_client
    if _claude_client is None:
        _claude_client = ClaudeClient()
    return _claude_client


async def analyze_query(query: str) -> QueryIntelligence:
    """Enrich *query* with intent classification and term expansion via Claude Haiku.

    Gracefully degrades to a no-op fallback if the Anthropic API is
    unavailable or returns unexpected output.

    Args:
        query: Raw user search query.

    Returns:
        :class:`QueryIntelligence` with ``intent``, ``expanded_query``,
        and ``rewritten_query``.
    """
    client = _get_client()
    intelligence = await client.analyze_query(query)
    logger.info(
        "intel.query_analyzed",
        intent=intelligence.intent,
        expanded=intelligence.expanded_query[:80],
    )
    return intelligence
