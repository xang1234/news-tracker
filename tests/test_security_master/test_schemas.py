"""Tests for security master schemas."""

from datetime import datetime, timezone

from src.security_master.schemas import Security


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
        now = datetime.now(timezone.utc)
        sec = Security(
            ticker="005930.KS",
            exchange="KRX",
            name="Samsung Electronics",
            aliases=["samsung", "samsung semiconductor"],
            sector="memory",
            country="KR",
            currency="KRW",
            figi="BBG000H7TBB4",
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        assert sec.ticker == "005930.KS"
        assert sec.exchange == "KRX"
        assert sec.aliases == ["samsung", "samsung semiconductor"]
        assert sec.country == "KR"
        assert sec.currency == "KRW"

    def test_aliases_mutable_default(self):
        """Ensure each instance gets its own aliases list."""
        a = Security(ticker="A")
        b = Security(ticker="B")
        a.aliases.append("test")
        assert b.aliases == []
