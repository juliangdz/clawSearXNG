"""Stage 1 — Query intelligence (tiered local rules or Claude Haiku)."""

from __future__ import annotations

import re

import structlog

from ai_search.config import settings
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


_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "biomedical": (
        "pubmed",
        "clinical",
        "disease",
        "patient",
        "hospital",
        "pathology",
        "tb",
        "tuberculosis",
        "cancer",
        "biomedical",
        "medical",
    ),
    "code": (
        "github",
        "repo",
        "implementation",
        "code",
        "python",
        "javascript",
        "docker",
        "api",
        "sdk",
    ),
    "research": (
        "paper",
        "arxiv",
        "citation",
        "scholar",
        "survey",
        "journal",
        "conference",
        "study",
    ),
    "news": (
        "news",
        "today",
        "latest",
        "breaking",
        "update",
    ),
}

_EXPANSIONS: dict[str, str] = {
    "tb": "tuberculosis",
    "bp": "blood pressure",
    "hr": "heart rate",
    "spo2": "oxygen saturation",
    "ai": "artificial intelligence",
    "ml": "machine learning",
}


def _rewrite_query(query: str) -> str:
    """Normalize whitespace for display."""
    return " ".join(query.split())


def _detect_intent(query: str) -> str:
    """Classify intent with lightweight keyword scoring."""
    q = query.lower()
    scores = {k: 0 for k in _INTENT_KEYWORDS}

    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                scores[intent] += 1

    best_intent = max(scores, key=scores.get)
    return best_intent if scores[best_intent] > 0 else "general"


def _expand_query(query: str) -> str:
    """Expand known abbreviations/synonyms without external LLM calls."""
    rewritten = _rewrite_query(query)
    q = rewritten.lower()

    additions: list[str] = []
    for token, expansion in _EXPANSIONS.items():
        if re.search(rf"\b{re.escape(token)}\b", q):
            additions.append(expansion)

    if not additions:
        return rewritten

    # Deduplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for item in additions:
        if item not in seen and item not in q:
            seen.add(item)
            unique.append(item)

    return f"{rewritten} {' '.join(unique)}".strip()


def _analyze_tiered(query: str) -> QueryIntelligence:
    """Local zero-cost query intelligence (default mode)."""
    rewritten = _rewrite_query(query)
    return QueryIntelligence(
        intent=_detect_intent(rewritten),
        expanded_query=_expand_query(rewritten),
        rewritten_query=rewritten,
    )


async def analyze_query(query: str) -> QueryIntelligence:
    """Enrich *query* with mode-selectable query intelligence.

    Modes:
    - ``tiered`` (default): local rule-based intent + query expansion (no API cost)
    - ``haiku``: Claude Haiku intent + expansion
    """
    mode = settings.query_intelligence_mode.strip().lower()

    if mode == "haiku":
        client = _get_client()
        intelligence = await client.analyze_query(query)
        # If Haiku degrades to generic fallback, prefer local tiered analysis.
        if intelligence.intent == "general" and intelligence.expanded_query.strip() == query.strip():
            intelligence = _analyze_tiered(query)
            logger.info("intel.query_analyzed", mode="tiered_fallback", intent=intelligence.intent)
            return intelligence

        logger.info(
            "intel.query_analyzed",
            mode="haiku",
            intent=intelligence.intent,
            expanded=intelligence.expanded_query[:80],
        )
        return intelligence

    intelligence = _analyze_tiered(query)
    logger.info(
        "intel.query_analyzed",
        mode="tiered",
        intent=intelligence.intent,
        expanded=intelligence.expanded_query[:80],
    )
    return intelligence
