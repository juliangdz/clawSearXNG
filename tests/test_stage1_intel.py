"""Tests for Stage 1 query intelligence modes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_search.pipeline import stage1_intel


@pytest.mark.asyncio
async def test_tiered_mode_detects_biomedical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiered mode should classify biomedical-style queries locally."""
    monkeypatch.setattr(stage1_intel.settings, "query_intelligence_mode", "tiered")

    result = await stage1_intel.analyze_query("latest pubmed paper on tb pathology")

    assert result.intent == "biomedical"
    assert "tuberculosis" in result.expanded_query.lower()


@pytest.mark.asyncio
async def test_haiku_mode_uses_client_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Haiku mode should use Claude client output when non-fallback output is returned."""
    monkeypatch.setattr(stage1_intel.settings, "query_intelligence_mode", "haiku")

    fake_client = MagicMock()
    fake_client.analyze_query = AsyncMock(
        return_value=stage1_intel.QueryIntelligence(
            intent="research",
            expanded_query="transformer attention mechanism survey",
            rewritten_query="transformer attention mechanism",
        )
    )
    monkeypatch.setattr(stage1_intel, "_get_client", lambda: fake_client)

    result = await stage1_intel.analyze_query("transformer attention")

    assert result.intent == "research"
    assert "survey" in result.expanded_query


@pytest.mark.asyncio
async def test_haiku_mode_falls_back_to_tiered(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Haiku response is generic fallback-like, tiered analyzer should be used."""
    monkeypatch.setattr(stage1_intel.settings, "query_intelligence_mode", "haiku")

    query = "github repo for biomedical segmentation"
    fake_client = MagicMock()
    fake_client.analyze_query = AsyncMock(
        return_value=stage1_intel.QueryIntelligence(
            intent="general",
            expanded_query=query,
            rewritten_query=query,
        )
    )
    monkeypatch.setattr(stage1_intel, "_get_client", lambda: fake_client)

    result = await stage1_intel.analyze_query(query)

    assert result.intent in {"code", "biomedical"}
