"""Anthropic Claude Haiku client for query intelligence."""

from __future__ import annotations

import json
import re

import anthropic
import structlog

from ai_search.config import settings
from ai_search.models import QueryIntelligence

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a search query optimizer. Given a user query, return ONLY valid JSON with these fields:\n"
    "- intent: one of [research, biomedical, code, news, general]\n"
    "- expanded_query: improved version with synonyms, related terms, year range if relevant\n"
    "- rewritten_query: clean display version"
)

_VALID_INTENTS = frozenset({"research", "biomedical", "code", "news", "general"})


class ClaudeClient:
    """Wrapper around the Anthropic Python SDK for query analysis.

    Instantiated once and reused across requests (the SDK manages its own
    HTTP connection pool internally).
    """

    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def analyze_query(self, query: str) -> QueryIntelligence:
        """Ask Claude Haiku to classify and expand *query*.

        On any failure (API error, malformed JSON, missing fields) the method
        returns a safe fallback so the pipeline can continue without
        intelligence enrichment.

        Args:
            query: Raw user search query.

        Returns:
            :class:`QueryIntelligence` with intent, expanded_query, and rewritten_query.
        """
        try:
            message = await self._client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )
            raw_text = message.content[0].text.strip()

            # Strip any markdown code fences that the model might add.
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            data = json.loads(raw_text)

            intent = data.get("intent", "general")
            if intent not in _VALID_INTENTS:
                intent = "general"

            return QueryIntelligence(
                intent=intent,
                expanded_query=str(data.get("expanded_query", query)),
                rewritten_query=str(data.get("rewritten_query", query)),
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "claude.analyze_query_failed",
                query=query,
                error=str(exc),
            )
            return _fallback(query)


def _fallback(query: str) -> QueryIntelligence:
    """Return a safe no-op intelligence object."""
    return QueryIntelligence(
        intent="general",
        expanded_query=query,
        rewritten_query=query,
    )
