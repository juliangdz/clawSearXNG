"""Domain authority and engine trust weights for coarse scoring."""

from __future__ import annotations

HIGH_AUTHORITY: dict[str, float] = {
    "arxiv.org": 1.0,
    "pubmed.ncbi.nlm.nih.gov": 1.0,
    "ncbi.nlm.nih.gov": 0.95,
    "nature.com": 0.95,
    "science.org": 0.95,
    "nejm.org": 0.95,
    "thelancet.com": 0.95,
    "jamanetwork.com": 0.90,
    "semanticscholar.org": 0.90,
    "cell.com": 0.90,
    "bmj.com": 0.90,
    "paperswithcode.com": 0.88,
    "huggingface.co": 0.88,
    "github.com": 0.85,
    "openai.com": 0.85,
    "anthropic.com": 0.85,
    "deepmind.google": 0.85,
    "acm.org": 0.85,
    "ieee.org": 0.85,
    "springer.com": 0.80,
}

MEDIUM_AUTHORITY: dict[str, float] = {
    "towardsdatascience.com": 0.65,
    "medium.com": 0.55,
    "kdnuggets.com": 0.60,
    "stackoverflow.com": 0.75,
    "reddit.com": 0.50,
    "wikipedia.org": 0.70,
}

DEFAULT_AUTHORITY: float = 0.30

ENGINE_TRUST: dict[str, float] = {
    "arxiv": 1.0,
    "pubmed": 1.0,
    "semantic_scholar": 1.0,
    "github": 0.85,
    "ddg": 0.70,
    "brave": 0.70,
}

HEALTHCARE_ML_DOMAINS: frozenset[str] = frozenset(
    {
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "nature.com",
        "nejm.org",
        "thelancet.com",
        "jamanetwork.com",
        "semanticscholar.org",
        "paperswithcode.com",
        "huggingface.co",
        "github.com",
        "bmj.com",
        "cell.com",
        "ncbi.nlm.nih.gov",
        "science.org",
    }
)


def get_authority(domain: str) -> float:
    """Return the authority weight for *domain*.

    Args:
        domain: Lowercase hostname, e.g. ``"arxiv.org"``.

    Returns:
        Float in [0, 1] representing domain authority.
    """
    return HIGH_AUTHORITY.get(domain, MEDIUM_AUTHORITY.get(domain, DEFAULT_AUTHORITY))


def get_engine_trust(engine: str) -> float:
    """Return the trust weight for a SearXNG *engine* name.

    Args:
        engine: Engine name as reported by SearXNG.

    Returns:
        Float in [0, 1].
    """
    return ENGINE_TRUST.get(engine, 0.5)
