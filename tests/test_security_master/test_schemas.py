"""Tests for security master schemas."""

from datetime import UTC, datetime

import pytest

from src.security_master.schemas import Security, SecurityIdentifierLineage


class TestSecurity:
    """Tests for Security dataclass."""

    def test_defaults(self):
        sec = Security(ticker="NVDA")
        assert sec.ticker == "NVDA"
        assert sec.exchange == "US"
        assert sec.name == ""
        assert sec.aliases == []
        assert sec.sector == ""
        assert sec.country == "US"
        assert sec.currency == "USD"
        assert sec.figi is None
        assert sec.is_active is True
        assert sec.created_at is None
        assert sec.updated_at is None

    def test_full_construction(self):
        now = datetime.now(UTC)
        sec = Security(
            ticker="005930.KS",
            exchange="KRX",
            name="Samsung Electronics",
            aliases=["samsung", "samsung semiconductor"],
            sector="memory",
            country="KR",
            currency="KRW",
            figi="BBG000H7TBB4",
            sec_cik="0012345678",
            issuer_name="Samsung Electronics Co Ltd",
            former_names=["Samsung Electronics Co., Ltd."],
            external_identifiers={"lei": "549300EXAMPLE"},
            identifier_lineage=[
                SecurityIdentifierLineage(
                    identifier_type="sec_cik",
                    value="0012345678",
                    source="sec_ticker_company",
                    observed_at="2026-05-31",
                )
            ],
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        assert sec.ticker == "005930.KS"
        assert sec.exchange == "KRX"
        assert sec.aliases == ["samsung", "samsung semiconductor"]
        assert sec.country == "KR"
        assert sec.currency == "KRW"
        assert sec.sec_cik == "0012345678"
        assert sec.issuer_name == "Samsung Electronics Co Ltd"
        assert sec.former_names == ["Samsung Electronics Co., Ltd."]
        assert sec.external_identifiers == {"lei": "549300EXAMPLE"}
        assert sec.identifier_lineage[0].source == "sec_ticker_company"

    def test_sec_cik_normalized_to_ten_digits(self):
        sec = Security(ticker="AAPL", sec_cik="CIK320193")
        assert sec.sec_cik == "0000320193"

    def test_invalid_sec_cik_rejected(self):
        with pytest.raises(ValueError, match="sec_cik"):
            Security(ticker="BAD", sec_cik="not-a-cik")

    def test_lineage_requires_auditable_source_type_and_value(self):
        with pytest.raises(ValueError, match="identifier_lineage"):
            SecurityIdentifierLineage(
                identifier_type="",
                value="0000320193",
                source="sec_ticker_company",
            )

    def test_aliases_mutable_default(self):
        """Ensure each instance gets its own aliases list."""
        a = Security(ticker="A")
        b = Security(ticker="B")
        a.aliases.append("test")
        assert b.aliases == []

    def test_former_names_mutable_default(self):
        """Ensure each instance gets its own former names list."""
        a = Security(ticker="A")
        b = Security(ticker="B")
        a.former_names.append("Old A")
        assert b.former_names == []
