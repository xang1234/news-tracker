"""Tests for SecApiProvider — fallback SEC filing provider.

Uses httpx mock responses to avoid real SEC EDGAR network calls.
Tests verify provider contract compliance, error handling, and
lineage parity with the primary EdgarToolsProvider.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.filing.schemas import FilingIdentity, FilingResult, FilingSection
from src.filing.sec_api_provider import (
    SecApiProvider,
    _normalize_form_type,
    _parse_date_str,
    _section_id,
)
from src.filing.sec_policy import SECPolicy


# -- Pure function tests ---------------------------------------------------


class TestNormalizeFormType:
    """_normalize_form_type handles SEC form variations."""

    def test_standard_types(self) -> None:
        assert _normalize_form_type("10-K") == "10-K"
        assert _normalize_form_type("10-Q") == "10-Q"
        assert _normalize_form_type("8-K") == "8-K"

    def test_amendment_stripped(self) -> None:
        assert _normalize_form_type("10-K/A") == "10-K"

    def test_aliases(self) -> None:
        assert _normalize_form_type("10K") == "10-K"
        assert _normalize_form_type("DEF14A") == "DEF 14A"

    def test_unknown_defaults_to_8k(self) -> None:
        assert _normalize_form_type("UNKNOWN") == "8-K"


class TestParseDateStr:
    """_parse_date_str handles SEC API date formats."""

    def test_valid_iso(self) -> None:
        assert _parse_date_str("2024-03-15") == date(2024, 3, 15)

    def test_none_returns_today(self) -> None:
        assert _parse_date_str(None) == date.today()

    def test_empty_returns_today(self) -> None:
        assert _parse_date_str("") == date.today()

    def test_long_string_truncated(self) -> None:
        assert _parse_date_str("2024-03-15T10:30:00Z") == date(2024, 3, 15)


class TestSectionId:
    """_section_id determinism."""

    def test_deterministic(self) -> None:
        assert _section_id("acc", 0, "s") == _section_id("acc", 0, "s")

    def test_different_inputs(self) -> None:
        assert _section_id("acc", 0, "a") != _section_id("acc", 0, "b")


# -- Provider contract tests -----------------------------------------------


class TestSecApiProviderContract:
    """SecApiProvider satisfies FilingProvider interface."""

    def test_name(self) -> None:
        provider = SecApiProvider()
        assert provider.name == "sec_api"

    def test_has_policy(self) -> None:
        provider = SecApiProvider()
        assert isinstance(provider.policy, SECPolicy)

    def test_custom_policy(self) -> None:
        policy = SECPolicy(rate_limit_per_second=5)
        provider = SecApiProvider(policy=policy)
        assert provider.policy.rate_limit_per_second == 5


# -- fetch_filing tests (mocked HTTP) -------------------------------------


def _mock_client() -> AsyncMock:
    """Create a mock httpx client with is_closed=False."""
    client = AsyncMock()
    client.is_closed = False
    return client


class TestFetchFiling:
    """fetch_filing with mocked httpx responses."""

    async def test_fetch_success(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        index_resp = MagicMock(status_code=200)
        text_resp = MagicMock(status_code=200)
        text_resp.text = "Filing full text content with multiple words here."

        mock_client.get = AsyncMock(side_effect=[index_resp, text_resp])
        provider._client = mock_client

        result = await provider.fetch_filing("0001234567-24-000123")

        assert isinstance(result, FilingResult)
        assert result.provider == "sec_api"
        assert result.status == "parsed"
        assert len(result.sections) == 1
        assert result.sections[0].section_name == "Full Text"
        assert result.sections[0].word_count > 0

    async def test_fetch_text_not_found(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        index_resp = MagicMock(status_code=200)
        text_resp = MagicMock(status_code=404)

        mock_client.get = AsyncMock(side_effect=[index_resp, text_resp])
        provider._client = mock_client

        result = await provider.fetch_filing("0001234567-24-000123")

        assert result.status == "fetched"
        assert len(result.sections) == 0

    async def test_fetch_index_failed(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        index_resp = MagicMock(status_code=503)

        mock_client.get = AsyncMock(return_value=index_resp)
        provider._client = mock_client

        result = await provider.fetch_filing("0001234567-24-000123")

        assert result.status == "failed"
        assert "503" in result.error_message

    async def test_fetch_network_error(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        mock_client.get = AsyncMock(side_effect=ConnectionError("timeout"))
        provider._client = mock_client

        result = await provider.fetch_filing("acc-001")

        assert result.status == "failed"
        assert "timeout" in result.error_message

    async def test_fetch_preserves_lineage(self) -> None:
        """Failed results still carry accession number for audit."""
        provider = SecApiProvider()

        mock_client = _mock_client()
        mock_client.get = AsyncMock(side_effect=RuntimeError("error"))
        provider._client = mock_client

        result = await provider.fetch_filing("acc-123")

        assert result.identity.accession_number == "acc-123"
        assert result.provider == "sec_api"
        assert result.fetched_at is not None


# -- search_filings tests (mocked HTTP) -----------------------------------


class TestSearchFilings:
    """search_filings with mocked httpx responses."""

    async def test_search_success(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "entity_id": "0001234567",
                            "accession_no": "0001234567-24-000001",
                            "form_type": "10-K",
                            "file_date": "2024-03-15",
                            "entity_name": "Test Corp",
                        }
                    }
                ]
            }
        }
        mock_client.get = AsyncMock(return_value=resp)
        provider._client = mock_client

        results = await provider.search_filings(cik="0001234567")

        assert len(results) == 1
        assert isinstance(results[0], FilingIdentity)
        assert results[0].cik == "0001234567"
        assert results[0].filing_type == "10-K"

    async def test_search_with_ticker(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": {"hits": []}}
        mock_client.get = AsyncMock(return_value=resp)
        provider._client = mock_client

        results = await provider.search_filings(ticker="NVDA")

        assert results == []

    async def test_search_requires_cik_or_ticker(self) -> None:
        provider = SecApiProvider()
        with pytest.raises(ValueError, match="cik or ticker"):
            await provider.search_filings()

    async def test_search_http_error(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        resp = MagicMock()
        resp.status_code = 500
        mock_client.get = AsyncMock(return_value=resp)
        provider._client = mock_client

        results = await provider.search_filings(cik="001")

        assert results == []

    async def test_search_network_error(self) -> None:
        provider = SecApiProvider()

        mock_client = _mock_client()
        mock_client.get = AsyncMock(side_effect=ConnectionError("timeout"))
        provider._client = mock_client

        results = await provider.search_filings(cik="001")

        assert results == []

    async def test_search_respects_limit(self) -> None:
        provider = SecApiProvider()

        hits = [
            {
                "_source": {
                    "entity_id": f"cik_{i}",
                    "accession_no": f"acc-{i}",
                    "form_type": "10-K",
                    "file_date": "2024-01-01",
                    "entity_name": f"Corp {i}",
                }
            }
            for i in range(10)
        ]
        mock_client = _mock_client()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"hits": {"hits": hits}}
        mock_client.get = AsyncMock(return_value=resp)
        provider._client = mock_client

        results = await provider.search_filings(cik="001", limit=3)

        assert len(results) == 3


# -- Provider parity tests ------------------------------------------------


class TestProviderParity:
    """Verify fallback produces same output shapes as primary."""

    async def test_result_has_same_fields(self) -> None:
        """FilingResult from SecApiProvider has all contract fields."""
        provider = SecApiProvider()

        mock_client = _mock_client()
        index_resp = MagicMock()
        index_resp.status_code = 200
        text_resp = MagicMock()
        text_resp.status_code = 200
        text_resp.text = "Some text."
        mock_client.get = AsyncMock(side_effect=[index_resp, text_resp])
        provider._client = mock_client

        result = await provider.fetch_filing("acc-001")

        # Verify all contract fields are present
        assert hasattr(result, "identity")
        assert hasattr(result, "sections")
        assert hasattr(result, "raw_url")
        assert hasattr(result, "status")
        assert hasattr(result, "provider")
        assert hasattr(result, "fetched_at")
        assert hasattr(result, "metadata")

        # Identity fields
        assert hasattr(result.identity, "cik")
        assert hasattr(result.identity, "accession_number")
        assert hasattr(result.identity, "filing_type")
        assert hasattr(result.identity, "filed_date")

    async def test_section_ids_deterministic(self) -> None:
        """Same accession produces same section IDs across runs."""
        provider = SecApiProvider()

        mock_client = _mock_client()
        index_resp = MagicMock()
        index_resp.status_code = 200
        text_resp = MagicMock()
        text_resp.status_code = 200
        text_resp.text = "Content."

        # Run twice
        mock_client.get = AsyncMock(side_effect=[index_resp, text_resp])
        provider._client = mock_client
        r1 = await provider.fetch_filing("acc-001")

        mock_client.get = AsyncMock(side_effect=[index_resp, text_resp])
        provider._client = mock_client
        r2 = await provider.fetch_filing("acc-001")

        assert r1.sections[0].section_id == r2.sections[0].section_id
