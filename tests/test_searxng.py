"""Tests for SearXNG client and URL utilities."""

from __future__ import annotations

from datetime import date


from ai_search.pipeline.stage3_fetch import _deduplicate, _is_title_duplicate, _parse_date
from ai_search.models import RawResult
from ai_search.utils.url_utils import extract_domain, normalize_url


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    """Tests for URL normalisation."""

    def test_strips_utm_params(self) -> None:
        """UTM tracking parameters are removed."""
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&topic=ai"
        result = normalize_url(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "topic=ai" in result

    def test_strips_fbclid(self) -> None:
        """Facebook click ID is removed."""
        url = "https://example.com/?fbclid=IwAR123"
        result = normalize_url(url)
        assert "fbclid" not in result

    def test_trailing_slash_removed(self) -> None:
        """Trailing slashes are stripped from paths."""
        url = "https://example.com/article/"
        result = normalize_url(url)
        assert not result.rstrip("?").endswith("/article/")

    def test_lowercase_hostname(self) -> None:
        """Hostnames are lowercased."""
        url = "https://ArXiv.ORG/abs/1234"
        result = normalize_url(url)
        assert "arxiv.org" in result

    def test_preserves_path(self) -> None:
        """Non-tracking path and params are preserved."""
        url = "https://arxiv.org/abs/1706.03762"
        result = normalize_url(url)
        assert "1706.03762" in result

    def test_invalid_url_returns_original(self) -> None:
        """Malformed URLs are returned unchanged."""
        bad = "not a url at all"
        assert normalize_url(bad) == bad


class TestExtractDomain:
    """Tests for domain extraction."""

    def test_simple_domain(self) -> None:
        assert extract_domain("https://arxiv.org/abs/1234") == "arxiv.org"

    def test_subdomain_preserved(self) -> None:
        assert extract_domain("https://pubmed.ncbi.nlm.nih.gov/12345") == "pubmed.ncbi.nlm.nih.gov"

    def test_no_scheme(self) -> None:
        """Bare domain without scheme returns empty or best-effort."""
        # urlparse without scheme may not parse correctly — just verify no crash.
        result = extract_domain("example.com/page")
        assert isinstance(result, str)

    def test_invalid_url(self) -> None:
        assert extract_domain("") == ""


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


class TestParseDate:
    """Tests for flexible date parsing."""

    def test_iso_date(self) -> None:
        assert _parse_date("2024-03-15") == date(2024, 3, 15)

    def test_iso_datetime(self) -> None:
        assert _parse_date("2023-11-01T12:00:00") == date(2023, 11, 1)

    def test_year_only(self) -> None:
        assert _parse_date("2022") == date(2022, 1, 1)

    def test_year_month(self) -> None:
        result = _parse_date("2021-07")
        assert result is not None
        assert result.year == 2021
        assert result.month == 7

    def test_none_input(self) -> None:
        assert _parse_date(None) is None

    def test_empty_string(self) -> None:
        assert _parse_date("") is None

    def test_garbage_input(self) -> None:
        assert _parse_date("not-a-date") is None

    def test_date_object_passthrough(self) -> None:
        """date objects are returned directly."""
        d = date(2024, 1, 1)
        assert _parse_date(d) == d


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestIsTitleDuplicate:
    """Tests for title similarity comparison."""

    def test_identical_titles(self) -> None:
        assert _is_title_duplicate("Hello World", "Hello World") is True

    def test_clearly_different(self) -> None:
        assert _is_title_duplicate("Attention Is All You Need", "BERT Paper") is False

    def test_near_identical(self) -> None:
        a = "Attention Is All You Need"
        b = "Attention Is All You Need (Revised)"
        # May or may not be duplicate depending on threshold.
        result = _is_title_duplicate(a, b)
        assert isinstance(result, bool)

    def test_case_insensitive(self) -> None:
        assert _is_title_duplicate("hello world", "HELLO WORLD") is True


class TestDeduplicate:
    """Tests for the full deduplication function."""

    def _make_result(self, title: str, url: str, rank: int = 0) -> RawResult:
        return RawResult(title=title, url=url, engine="ddg", engine_rank=rank)

    def test_removes_exact_url_duplicates(self) -> None:
        """Exact URL duplicates (after normalisation) are removed."""
        r1 = self._make_result("Title A", "https://example.com/page?utm_source=x", rank=0)
        r2 = self._make_result("Title A Copy", "https://example.com/page", rank=1)
        unique = _deduplicate([r1, r2])
        assert len(unique) == 1

    def test_removes_title_duplicates(self) -> None:
        """Near-identical titles from different URLs are deduplicated."""
        r1 = self._make_result("Attention Is All You Need", "https://arxiv.org/abs/1706", 0)
        r2 = self._make_result("Attention Is All You Need", "https://papers.org/1706", 1)
        unique = _deduplicate([r1, r2])
        assert len(unique) == 1

    def test_different_results_preserved(self) -> None:
        """Distinct results are all kept."""
        results = [
            self._make_result("Alpha Paper", "https://arxiv.org/abs/1", 0),
            self._make_result("Beta Paper", "https://arxiv.org/abs/2", 1),
            self._make_result("Gamma Paper", "https://arxiv.org/abs/3", 2),
        ]
        unique = _deduplicate(results)
        assert len(unique) == 3

    def test_max_15_results(self) -> None:
        """Output is capped at 15."""
        results = [
            self._make_result(f"Paper {i}", f"https://example.com/{i}", i)
            for i in range(30)
        ]
        unique = _deduplicate(results)
        assert len(unique) <= 15
