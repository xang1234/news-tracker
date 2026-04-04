"""Tests for filing lane schemas."""

from datetime import date, datetime, timezone

import pytest

from src.filing.schemas import (
    VALID_FILING_STATUSES,
    VALID_FILING_TYPES,
    FilingIdentity,
    FilingResult,
    FilingSection,
)


class TestFilingIdentity:
    """FilingIdentity dataclass validation."""

    def test_minimal_valid(self) -> None:
        fid = FilingIdentity(
            cik="0001234567",
            accession_number="0001234567-24-000123",
            filing_type="10-K",
            filed_date=date(2024, 3, 15),
        )
        assert fid.company_name == ""
        assert fid.ticker is None
        assert fid.period_of_report is None

    def test_all_filing_types_accepted(self) -> None:
        for ft in VALID_FILING_TYPES:
            fid = FilingIdentity(
                cik="001",
                accession_number="acc-001",
                filing_type=ft,
                filed_date=date(2024, 1, 1),
            )
            assert fid.filing_type == ft

    def test_invalid_filing_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid filing_type"):
            FilingIdentity(
                cik="001",
                accession_number="acc-001",
                filing_type="INVALID",
                filed_date=date(2024, 1, 1),
            )

    def test_full_identity(self) -> None:
        fid = FilingIdentity(
            cik="0000320193",
            accession_number="0000320193-24-000123",
            filing_type="10-K",
            filed_date=date(2024, 11, 1),
            period_of_report=date(2024, 9, 28),
            company_name="Apple Inc",
            ticker="AAPL",
        )
        assert fid.company_name == "Apple Inc"
        assert fid.ticker == "AAPL"


class TestFilingSection:
    """FilingSection dataclass."""

    def test_minimal_valid(self) -> None:
        s = FilingSection(section_id="s1", section_name="Risk Factors")
        assert s.section_type == "narrative"
        assert s.word_count == 0

    def test_with_content(self) -> None:
        s = FilingSection(
            section_id="s1",
            section_name="Risk Factors",
            content="We face significant risks...",
            word_count=5,
        )
        assert s.word_count == 5


class TestFilingResult:
    """FilingResult dataclass validation."""

    def _make_identity(self) -> FilingIdentity:
        return FilingIdentity(
            cik="001",
            accession_number="acc-001",
            filing_type="10-K",
            filed_date=date(2024, 1, 1),
        )

    def test_minimal_valid(self) -> None:
        r = FilingResult(identity=self._make_identity())
        assert r.status == "fetched"
        assert r.sections == []
        assert r.provider == ""

    def test_all_statuses_accepted(self) -> None:
        for s in VALID_FILING_STATUSES:
            r = FilingResult(identity=self._make_identity(), status=s)
            assert r.status == s

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid filing status"):
            FilingResult(identity=self._make_identity(), status="bad")

    def test_total_word_count(self) -> None:
        r = FilingResult(
            identity=self._make_identity(),
            sections=[
                FilingSection(section_id="s1", section_name="A", word_count=100),
                FilingSection(section_id="s2", section_name="B", word_count=200),
            ],
        )
        assert r.total_word_count == 300

    def test_section_names(self) -> None:
        r = FilingResult(
            identity=self._make_identity(),
            sections=[
                FilingSection(section_id="s1", section_name="Risk Factors"),
                FilingSection(section_id="s2", section_name="MD&A"),
            ],
        )
        assert r.section_names == ["Risk Factors", "MD&A"]

    def test_failed_result(self) -> None:
        r = FilingResult(
            identity=self._make_identity(),
            status="failed",
            error_message="Connection timeout",
            provider="edgartools",
        )
        assert r.error_message == "Connection timeout"


class TestFilingTypeCompleteness:
    """Verify filing types cover key SEC forms."""

    def test_annual_reports(self) -> None:
        assert "10-K" in VALID_FILING_TYPES

    def test_quarterly_reports(self) -> None:
        assert "10-Q" in VALID_FILING_TYPES

    def test_current_reports(self) -> None:
        assert "8-K" in VALID_FILING_TYPES

    def test_proxy_statements(self) -> None:
        assert "DEF 14A" in VALID_FILING_TYPES

    def test_foreign_filers(self) -> None:
        assert "20-F" in VALID_FILING_TYPES
        assert "6-K" in VALID_FILING_TYPES

    def test_ownership(self) -> None:
        assert "4" in VALID_FILING_TYPES
        assert "SC 13D" in VALID_FILING_TYPES

    def test_institutional_holdings(self) -> None:
        assert "13F-HR" in VALID_FILING_TYPES
