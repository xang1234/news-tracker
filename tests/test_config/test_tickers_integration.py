"""Tests for security master integration in tickers.py."""

from unittest.mock import AsyncMock

import pytest

from src.config import tickers


@pytest.fixture(autouse=True)
def _reset_tickers_cache():
    """Ensure each test starts with a clean cache."""
    tickers._reset_cache()
    yield
    tickers._reset_cache()


class TestStaticFallback:
    """When security master is NOT initialized, static dicts are used."""

    def test_get_all_tickers_returns_static(self):
        result = tickers.get_all_tickers()
        assert "NVDA" in result
        assert "AMD" in result

    def test_normalize_ticker_uses_static(self):
        assert tickers.normalize_ticker("NVDA") == "NVDA"
        assert tickers.normalize_ticker("$AMD") == "AMD"
        assert tickers.normalize_ticker("AAPL") is None

    def test_company_to_ticker_uses_static(self):
        assert tickers.company_to_ticker("nvidia") == "NVDA"
        assert tickers.company_to_ticker("unknown corp") is None


class TestCachedMode:
    """When module-level caches are populated, they take precedence."""

    def test_get_all_tickers_uses_cache(self):
        tickers._cached_tickers = {"AAPL", "GOOG"}

        result = tickers.get_all_tickers()

        assert result == {"AAPL", "GOOG"}
        assert "NVDA" not in result

    def test_get_all_tickers_returns_copy(self):
        tickers._cached_tickers = {"AAPL"}
        result = tickers.get_all_tickers()
        result.add("MUTATION")
        assert "MUTATION" not in tickers._cached_tickers

    def test_normalize_ticker_uses_cache(self):
        tickers._cached_tickers = {"AAPL"}

        assert tickers.normalize_ticker("AAPL") == "AAPL"
        assert tickers.normalize_ticker("NVDA") is None  # not in cache

    def test_company_to_ticker_uses_cache(self):
        tickers._cached_company_map = {"apple inc": "AAPL"}

        assert tickers.company_to_ticker("Apple Inc") == "AAPL"
        assert tickers.company_to_ticker("nvidia") is None  # not in cache


class TestInitSecurityMaster:
    """Tests for the async init_security_master function."""

    @pytest.mark.asyncio
    async def test_populates_caches(self):
        mock_db = AsyncMock()
        # count returns 35 (skips seeding), then ticker/map fetches
        mock_db.fetchval = AsyncMock(return_value=35)
        mock_db.fetch = AsyncMock(
            side_effect=[
                # First call: get_all_active_tickers
                [{"ticker": "NVDA"}, {"ticker": "TSM"}],
                # Second call: get_company_to_ticker_map
                [
                    {"ticker": "NVDA", "name": "NVIDIA", "aliases": ["nvidia"]},
                    {"ticker": "TSM", "name": "TSMC", "aliases": ["tsmc"]},
                ],
            ]
        )

        await tickers.init_security_master(mock_db)

        assert tickers._cached_tickers == {"NVDA", "TSM"}
        assert tickers._cached_company_map["nvidia"] == "NVDA"
        assert tickers._cached_company_map["tsmc"] == "TSM"

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        mock_db = AsyncMock()
        mock_db.fetchval = AsyncMock(side_effect=RuntimeError("DB down"))

        await tickers.init_security_master(mock_db)

        # Caches should remain None (fallback to static)
        assert tickers._cached_tickers is None
        assert tickers._cached_company_map is None

    @pytest.mark.asyncio
    async def test_static_still_works_after_failed_init(self):
        mock_db = AsyncMock()
        mock_db.fetchval = AsyncMock(side_effect=RuntimeError("DB down"))

        await tickers.init_security_master(mock_db)

        # Static fallback should still work
        assert "NVDA" in tickers.get_all_tickers()
        assert tickers.company_to_ticker("nvidia") == "NVDA"


class TestResetCache:
    """Tests for _reset_cache helper."""

    def test_clears_both_caches(self):
        tickers._cached_tickers = {"X"}
        tickers._cached_company_map = {"x": "X"}

        tickers._reset_cache()

        assert tickers._cached_tickers is None
        assert tickers._cached_company_map is None
