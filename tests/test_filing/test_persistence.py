"""Tests for filing persistence schemas and conversion."""

import pathlib
from datetime import date, datetime, timezone

import pytest

from src.filing.persistence import (
    FilingAttachmentRecord,
    FilingRecord,
    FilingSectionRecord,
    XBRLFactRecord,
    filing_result_to_records,
)
from src.filing.schemas import FilingIdentity, FilingResult, FilingSection

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "migrations"
    / "022_filing_artifacts.sql"
)


# -- Schema tests ----------------------------------------------------------


class TestFilingRecord:
    """FilingRecord dataclass validation."""

    def test_minimal_valid(self) -> None:
        r = FilingRecord(
            accession_number="0001234567-24-000123",
            cik="0001234567",
            filing_type="10-K",
            filed_date=date(2024, 3, 15),
        )
        assert r.status == "fetched"
        assert r.total_word_count == 0

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid filing status"):
            FilingRecord(
                accession_number="acc",
                cik="001",
                filing_type="10-K",
                filed_date=date(2024, 1, 1),
                status="bad",
            )

    def test_full_record(self) -> None:
        r = FilingRecord(
            accession_number="acc-001",
            cik="001",
            filing_type="10-K",
            filed_date=date(2024, 3, 15),
            company_name="Test Corp",
            ticker="TST",
            content_hash="sha256:abc",
            total_word_count=5000,
            section_count=10,
            provider="edgartools",
            run_id="run_001",
            status="parsed",
        )
        assert r.section_count == 10
        assert r.provider == "edgartools"


class TestFilingSectionRecord:
    """FilingSectionRecord dataclass."""

    def test_minimal_valid(self) -> None:
        s = FilingSectionRecord(
            section_id="sec_abc",
            accession_number="acc-001",
            section_index=0,
            section_name="Risk Factors",
        )
        assert s.section_type == "narrative"
        assert s.word_count == 0


class TestXBRLFactRecord:
    """XBRLFactRecord dataclass."""

    def test_minimal_valid(self) -> None:
        f = XBRLFactRecord(
            accession_number="acc-001",
            concept_name="Revenue",
            value="1000000000",
        )
        assert f.taxonomy == "us-gaap"
        assert f.unit is None

    def test_full_fact(self) -> None:
        f = XBRLFactRecord(
            accession_number="acc-001",
            concept_name="EarningsPerShareBasic",
            value="3.45",
            taxonomy="us-gaap",
            unit="USD/shares",
            period_start=date(2024, 1, 1),
            period_end=date(2024, 12, 31),
            decimals=2,
        )
        assert f.period_end == date(2024, 12, 31)


# -- Conversion tests ------------------------------------------------------


class TestFilingResultToRecords:
    """filing_result_to_records conversion."""

    def _make_result(self, **kwargs) -> FilingResult:
        defaults = dict(
            identity=FilingIdentity(
                cik="0001234567",
                accession_number="0001234567-24-000123",
                filing_type="10-K",
                filed_date=date(2024, 3, 15),
                company_name="Test Corp",
                ticker="TST",
            ),
            sections=[
                FilingSection(
                    section_id="sec_001",
                    section_name="Risk Factors",
                    content="We face significant risks...",
                    word_count=5,
                ),
                FilingSection(
                    section_id="sec_002",
                    section_name="MD&A",
                    content="Revenue increased 20%...",
                    word_count=4,
                ),
            ],
            raw_url="https://sec.gov/filing",
            status="parsed",
            provider="edgartools",
            fetched_at=datetime(2024, 3, 16, tzinfo=timezone.utc),
        )
        defaults.update(kwargs)
        return FilingResult(**defaults)

    def test_produces_filing_record(self) -> None:
        result = self._make_result()
        filing, sections = filing_result_to_records(result, run_id="run_001")
        assert filing.accession_number == "0001234567-24-000123"
        assert filing.cik == "0001234567"
        assert filing.filing_type == "10-K"
        assert filing.company_name == "Test Corp"
        assert filing.provider == "edgartools"
        assert filing.run_id == "run_001"
        assert filing.total_word_count == 9
        assert filing.section_count == 2

    def test_produces_section_records(self) -> None:
        result = self._make_result()
        _, sections = filing_result_to_records(result)
        assert len(sections) == 2
        assert sections[0].section_name == "Risk Factors"
        assert sections[0].section_index == 0
        assert sections[1].section_name == "MD&A"
        assert sections[1].section_index == 1

    def test_content_hash_computed(self) -> None:
        result = self._make_result()
        filing, _ = filing_result_to_records(result)
        assert filing.content_hash is not None
        assert filing.content_hash.startswith("sha256:")

    def test_section_hashes_computed(self) -> None:
        result = self._make_result()
        _, sections = filing_result_to_records(result)
        for s in sections:
            assert s.content_hash is not None
            assert s.content_hash.startswith("sha256:")

    def test_empty_sections(self) -> None:
        result = self._make_result(sections=[])
        filing, sections = filing_result_to_records(result)
        assert filing.section_count == 0
        assert filing.content_hash is None
        assert sections == []

    def test_ingested_at_from_fetched_at(self) -> None:
        result = self._make_result()
        filing, _ = filing_result_to_records(result)
        assert filing.ingested_at == datetime(2024, 3, 16, tzinfo=timezone.utc)

    def test_deterministic_hashes(self) -> None:
        r1 = self._make_result()
        r2 = self._make_result()
        f1, _ = filing_result_to_records(r1)
        f2, _ = filing_result_to_records(r2)
        assert f1.content_hash == f2.content_hash


# -- Migration tests -------------------------------------------------------


class TestMigration022:
    """Migration 022 structural checks."""

    @pytest.fixture()
    def sql(self) -> str:
        return MIGRATION_PATH.read_text()

    def test_file_exists(self) -> None:
        assert MIGRATION_PATH.exists()

    def test_creates_filings(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS filings" in sql

    def test_creates_filing_sections(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS filing_sections" in sql

    def test_creates_filing_attachments(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS filing_attachments" in sql

    def test_creates_filing_xbrl_facts(self, sql: str) -> None:
        assert "CREATE TABLE IF NOT EXISTS filing_xbrl_facts" in sql

    def test_accession_number_pk(self, sql: str) -> None:
        assert "accession_number    TEXT PRIMARY KEY" in sql

    def test_concept_id_fk(self, sql: str) -> None:
        assert "REFERENCES concepts(concept_id)" in sql

    def test_bitemporal_fields(self, sql: str) -> None:
        assert "source_published_at" in sql
        assert "ingested_at" in sql

    def test_content_hash_field(self, sql: str) -> None:
        assert "content_hash" in sql

    def test_status_check_constraint(self, sql: str) -> None:
        assert "'pending'" in sql
        assert "'parsed'" in sql
        assert "'failed'" in sql

    def test_updated_at_trigger(self, sql: str) -> None:
        assert "update_filings_updated_at" in sql

    def test_uses_if_not_exists(self, sql: str) -> None:
        for line in sql.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("CREATE TABLE ") and "IF NOT EXISTS" not in stripped:
                pytest.fail(f"CREATE TABLE without IF NOT EXISTS: {line.strip()}")
