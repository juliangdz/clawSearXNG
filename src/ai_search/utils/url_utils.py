"""URL normalisation helpers."""

from __future__ import annotations

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Query parameters that are purely tracking / analytics noise.
_STRIP_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "fbclid",
        "gclid",
        "msclkid",
        "ref",
        "referrer",
        "source",
        "_ga",
        "mc_cid",
        "mc_eid",
    }
)


def normalize_url(url: str) -> str:
    """Return a canonical form of *url* with tracking params and trailing slashes removed.

    Args:
        url: Raw URL string.

    Returns:
        Normalised URL suitable for deduplication comparison.
    """
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query, keep_blank_values=False)
        cleaned_qs = {k: v for k, v in qs.items() if k not in _STRIP_PARAMS}
        clean_query = urlencode(cleaned_qs, doseq=True)
        path = parsed.path.rstrip("/") or "/"
        normalised = urlunparse(
            (parsed.scheme, parsed.netloc.lower(), path, parsed.params, clean_query, "")
        )
        return normalised
    except Exception:  # noqa: BLE001
        return url


def extract_domain(url: str) -> str:
    """Return the bare hostname (no port) for *url*.

    Args:
        url: Any URL string.

    Returns:
        Lowercase hostname, e.g. ``"arxiv.org"``.
    """
    try:
        return urlparse(url).hostname or ""
    except Exception:  # noqa: BLE001
        return ""
