"""Tests for EdgarToolsProvider.

Uses mocked edgartools objects to avoid SEC EDGAR network calls.
Validates that the provider correctly translates edgartools objects
into our FilingResult contract shape.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.filing.edgartools_provider import (
    EdgarToolsProvider,
    _extract_sections,
    _filing_to_result,
)
from src.filing.schemas import FilingIdentity, FilingResult
from src.filing.sec_policy import SECPolicy
from src.filing.utils import (
    make_section_id,
    normalize_filing_type,
    parse_filing_date,
)

# -- Helper: mock edgartools Filing object ---------------------------------


def _mock_filing(
    *,
    accession: str = "0001234567-24-000123",
    cik: str = "0001234567",
    form_type: str = "10-K",
    filed: str = "2024-03-15",
    company_name: str = "Test Corp",
    period_of_report: str = "2024-12-31",
    sections: list | None = None,
    text: str = "Filing full text content here.",
) -> MagicMock:
    """Create a mock edgartools Filing."""
    filing = MagicMock()
    filing.accession_number = accession
    filing.cik = cik
    filing.period_of_report = period_of_report
    filing.filing_url = f"https://sec.gov/Archives/edgar/data/{cik}/{accession}"
    filing.url = filing.filing_url

    header = SimpleNamespace(
        cik=cik,
        form_type=form_type,
        filed=filed,
        company_name=company_name,
    )
    filing.header = header

    if sections is not None:
        filing.sections = sections
    else:
        filing.sections = None

    filing.text = MagicMock(return_value=text)
    return filing


def _mock_section(name: str, content: str) -> SimpleNamespace:
    """Create a mock edgartools section."""
    return SimpleNamespace(title=name, name=name, __str__=lambda self: content)


# -- Pure function tests ---------------------------------------------------


class TestNormalizeFilingType:
    """normalize_filing_type() handles SEC form variations."""

    def test_standard_types(self) -> None:
        assert normalize_filing_type("10-K") == "10-K"
        assert normalize_filing_type("10-Q") == "10-Q"
        assert normalize_filing_type("8-K") == "8-K"

    def test_amendment_suffix_stripped(self) -> None:
        assert normalize_filing_type("10-K/A") == "10-K"
        assert normalize_filing_type("10-Q/A") == "10-Q"
        assert normalize_filing_type("8-K/A") == "8-K"

    def test_case_insensitive(self) -> None:
        assert normalize_filing_type("10-k") == "10-K"
        assert normalize_filing_type("def 14a") == "DEF 14A"

    def test_aliases(self) -> None:
        assert normalize_filing_type("10K") == "10-K"
        assert normalize_filing_type("10Q") == "10-Q"
        assert normalize_filing_type("DEF14A") == "DEF 14A"
        assert normalize_filing_type("FORM 4") == "4"

    def test_unrecognized_defaults_to_8k(self) -> None:
        assert normalize_filing_type("UNKNOWN-FORM") == "8-K"

    def test_whitespace_handled(self) -> None:
        assert normalize_filing_type("  10-K  ") == "10-K"


class TestParseDate:
    """parse_filing_date() handles edgartools date formats."""

    def test_date_object(self) -> None:
        d = date(2024, 3, 15)
        assert parse_filing_date(d) == d

    def test_datetime_object(self) -> None:
        dt = datetime(2024, 3, 15, 10, 30, tzinfo=UTC)
        assert parse_filing_date(dt) == date(2024, 3, 15)

    def test_iso_string(self) -> None:
        assert parse_filing_date("2024-03-15") == date(2024, 3, 15)

    def test_none_returns_today(self) -> None:
        assert parse_filing_date(None) == date.today()


class TestSectionId:
    """make_section_id() is deterministic."""

    def test_deterministic(self) -> None:
        id1 = make_section_id("acc-001", 0, "Risk Factors")
        id2 = make_section_id("acc-001", 0, "Risk Factors")
        assert id1 == id2

    def test_different_inputs(self) -> None:
        id1 = make_section_id("acc-001", 0, "Risk Factors")
        id2 = make_section_id("acc-001", 1, "MD&A")
        assert id1 != id2

    def test_format(self) -> None:
        sid = make_section_id("acc-001", 0, "Risk Factors")
        assert sid.startswith("sec_")


class TestExtractSections:
    """_extract_sections() handles various filing structures."""

    def test_from_sections_attribute(self) -> None:
        sections = [
            _mock_section("Risk Factors", "We face risks..."),
            _mock_section("MD&A", "Revenue grew..."),
        ]
        filing = _mock_filing(sections=sections)
        result = _extract_sections(filing, "acc-001")
        assert len(result) == 2
        assert result[0].section_name == "Risk Factors"
        assert result[1].section_name == "MD&A"

    def test_fallback_to_full_text(self) -> None:
        filing = _mock_filing(text="Full filing text here with many words.")
        result = _extract_sections(filing, "acc-001")
        assert len(result) == 1
        assert result[0].section_name == "Full Text"
        assert result[0].word_count > 0

    def test_empty_filing(self) -> None:
        filing = _mock_filing(text="")
        result = _extract_sections(filing, "acc-001")
        assert len(result) == 0


class TestFilingToResult:
    """_filing_to_result() translates edgartools to FilingResult."""

    def test_basic_conversion(self) -> None:
        filing = _mock_filing()
        result = _filing_to_result(filing, "edgartools")
        assert isinstance(result, FilingResult)
        assert result.provider == "edgartools"
        assert result.identity.cik == "0001234567"
        assert result.identity.filing_type == "10-K"
        assert result.identity.filed_date == date(2024, 3, 15)
        assert result.identity.company_name == "Test Corp"

    def test_sections_extracted(self) -> None:
        filing = _mock_filing(text="Some content with words.")
        result = _filing_to_result(filing, "edgartools")
        assert len(result.sections) >= 1

    def test_url_populated(self) -> None:
        filing = _mock_filing()
        result = _filing_to_result(filing, "edgartools")
        assert "sec.gov" in result.raw_url

    def test_fetched_at_set(self) -> None:
        filing = _mock_filing()
        result = _filing_to_result(filing, "edgartools")
        assert result.fetched_at is not None


# -- Provider integration tests (mocked) ----------------------------------


class TestEdgarToolsProvider:
    """EdgarToolsProvider with mocked edgartools."""

    def test_name(self) -> None:
        provider = EdgarToolsProvider()
        assert provider.name == "edgartools"

    def test_has_policy(self) -> None:
        provider = EdgarToolsProvider()
        assert isinstance(provider.policy, SECPolicy)

    @patch("src.filing.edgartools_provider._ensure_edgartools")
    async def test_fetch_filing_success(self, mock_edgar) -> None:
        mock_module = MagicMock()
        mock_filing_obj = _mock_filing()
        mock_module.find.return_value = mock_filing_obj
        mock_edgar.return_value = mock_module

        provider = EdgarToolsProvider()
        result = await provider.fetch_filing("0001234567-24-000123")

        assert isinstance(result, FilingResult)
        assert result.identity.accession_number == "0001234567-24-000123"
        assert result.provider == "edgartools"

    @patch("src.filing.edgartools_provider._ensure_edgartools")
    async def test_fetch_filing_not_found(self, mock_edgar) -> None:
        mock_module = MagicMock()
        mock_module.find.return_value = None
        mock_edgar.return_value = mock_module

        provider = EdgarToolsProvider()
        result = await provider.fetch_filing("nonexistent")

        assert result.status == "failed"
        assert "not found" in result.error_message.lower()

    @patch("src.filing.edgartools_provider._ensure_edgartools")
    async def test_fetch_filing_error_handled(self, mock_edgar) -> None:
        mock_module = MagicMock()
        mock_module.find.side_effect = RuntimeError("Network error")
        mock_edgar.return_value = mock_module

        provider = EdgarToolsProvider()
        result = await provider.fetch_filing("acc-001")

        assert result.status == "failed"
        assert "Network error" in result.error_message

    @patch("src.filing.edgartools_provider._ensure_edgartools")
    async def test_search_filings_success(self, mock_edgar) -> None:
        mock_module = MagicMock()
        mock_company = MagicMock()
        mock_filing_obj = _mock_filing(accession="acc-001", form_type="10-K", filed="2024-06-15")
        mock_company.get_filings.return_value = [mock_filing_obj]
        mock_company.cik = "0001234567"
        mock_module.Company.return_value = mock_company
        mock_edgar.return_value = mock_module

        provider = EdgarToolsProvider()
        results = await provider.search_filings(ticker="NVDA", limit=5)

        assert len(results) == 1
        assert isinstance(results[0], FilingIdentity)
        assert results[0].accession_number == "acc-001"

    async def test_search_requires_cik_or_ticker(self) -> None:
        provider = EdgarToolsProvider()
        with pytest.raises(ValueError, match="cik or ticker"):
            await provider.search_filings()

    @patch("src.filing.edgartools_provider._ensure_edgartools")
    async def test_search_error_returns_empty(self, mock_edgar) -> None:
        mock_module = MagicMock()
        mock_module.Company.side_effect = RuntimeError("Lookup failed")
        mock_edgar.return_value = mock_module

        provider = EdgarToolsProvider()
        results = await provider.search_filings(cik="001")

        assert results == []
