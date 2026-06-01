"""Tests for SecurityMasterRepository."""

import json
from unittest.mock import AsyncMock

import pytest

from src.security_master.repository import SecurityMasterRepository
from src.security_master.schemas import Security


class TestUpsert:
    """Tests for single-security upsert."""

    @pytest.mark.asyncio
    async def test_passes_correct_params(
        self, mock_database: AsyncMock, sample_security: Security
    ) -> None:
        repo = SecurityMasterRepository(mock_database)
        mock_database.fetch.return_value = [{"ticker": "NVDA", "exchange": "US"}]

        await repo.upsert(sample_security)

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "INSERT INTO securities" in sql
        assert "ON CONFLICT (ticker, exchange) DO UPDATE" in sql
        assert args[1] == "NVDA"
        assert args[2] == "US"
        assert args[3] == "NVIDIA Corporation"
        assert args[4] == ["nvidia", "nvda", "geforce", "jensen huang"]
        assert args[9] == "0001045810"
        assert args[10] == "NVIDIA Corporation"
        assert args[11] == []
        assert json.loads(args[12]) == {"sec_ticker": "NVDA"}
        lineage = json.loads(args[13])
        assert lineage[0]["identifier_type"] == "sec_cik"
        assert lineage[0]["value"] == "0001045810"
        assert lineage[0]["source"] == "sec_ticker_company"
        assert lineage[0]["observed_at"] == "2026-05-31"

    @pytest.mark.asyncio
    async def test_upsert_with_figi(self, mock_database: AsyncMock) -> None:
        repo = SecurityMasterRepository(mock_database)
        sec = Security(ticker="TSM", name="TSMC", figi="BBG000BD8ZK0")
        mock_database.fetch.return_value = [{"ticker": "TSM", "exchange": "US"}]

        await repo.upsert(sec)

        args = mock_database.fetch.call_args[0]
        assert args[8] == "BBG000BD8ZK0"  # figi param


class TestBulkUpsert:
    """Tests for bulk upsert."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zero(self, mock_database: AsyncMock) -> None:
        repo = SecurityMasterRepository(mock_database)
        result = await repo.bulk_upsert([])
        assert result == 0
        mock_database.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_parallel_arrays(self, mock_database: AsyncMock) -> None:
        repo = SecurityMasterRepository(mock_database)
        securities = [
            Security(ticker="NVDA", name="NVIDIA"),
            Security(ticker="AMD", name="AMD Inc"),
        ]

        result = await repo.bulk_upsert(securities)

        assert result == 2
        args = mock_database.execute.call_args[0]
        sql = args[0]
        assert "unnest" in sql
        assert args[1] == ["NVDA", "AMD"]  # tickers array
        assert args[3] == ["NVIDIA", "AMD Inc"]  # names array
        assert args[9] == [None, None]  # sec_cik array


class TestGetByTicker:
    """Tests for single-ticker lookup."""

    @pytest.mark.asyncio
    async def test_found(self, mock_database: AsyncMock, sample_db_row: dict) -> None:
        mock_database.fetchrow.return_value = sample_db_row
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_by_ticker("NVDA")

        assert result is not None
        assert result.ticker == "NVDA"
        assert result.name == "NVIDIA Corporation"
        assert result.aliases == ["nvidia", "nvda", "geforce", "jensen huang"]
        assert result.sec_cik == "0001045810"
        assert result.issuer_name == "NVIDIA Corporation"
        assert result.external_identifiers == {"sec_ticker": "NVDA"}
        assert result.identifier_lineage[0].source == "sec_ticker_company"

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        mock_database.fetchrow.return_value = None
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_by_ticker("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_with_exchange(self, mock_database: AsyncMock, sample_korean_row: dict) -> None:
        mock_database.fetchrow.return_value = sample_korean_row
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_by_ticker("005930.KS", exchange="KRX")

        assert result is not None
        assert result.exchange == "KRX"
        assert result.currency == "KRW"


class TestGetAllActive:
    """Tests for fetching all active securities."""

    @pytest.mark.asyncio
    async def test_returns_list(self, mock_database: AsyncMock, sample_db_row: dict) -> None:
        mock_database.fetch.return_value = [sample_db_row]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_all_active()

        assert len(result) == 1
        assert result[0].ticker == "NVDA"
        sql = mock_database.fetch.call_args[0][0]
        assert "is_active = TRUE" in sql

    @pytest.mark.asyncio
    async def test_empty(self, mock_database: AsyncMock) -> None:
        repo = SecurityMasterRepository(mock_database)
        result = await repo.get_all_active()
        assert result == []


class TestGetAllActiveTickers:
    """Tests for ticker-only fetch."""

    @pytest.mark.asyncio
    async def test_returns_set(self, mock_database: AsyncMock) -> None:
        mock_database.fetch.return_value = [
            {"ticker": "NVDA"},
            {"ticker": "AMD"},
            {"ticker": "TSM"},
        ]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_all_active_tickers()

        assert result == {"NVDA", "AMD", "TSM"}


class TestListSecurities:
    """Tests for paginated security listing."""

    @pytest.mark.asyncio
    async def test_search_matches_alias_and_former_name_values(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        mock_database.fetchval.return_value = 1
        mock_database.fetch.return_value = [sample_db_row]
        repo = SecurityMasterRepository(mock_database)

        result, total = await repo.list_securities(search="GeForce")

        sql = mock_database.fetch.call_args[0][0]
        assert "unnest(aliases) AS alias_value" in sql
        assert "alias_value ILIKE $1" in sql
        assert "unnest(former_names) AS former_name_value" in sql
        assert "former_name_value ILIKE $1" in sql
        assert "$1 ILIKE ANY(aliases)" not in sql
        assert "$1 ILIKE ANY(former_names)" not in sql
        assert total == 1
        assert result[0].ticker == "NVDA"


class TestGetCompanyToTickerMap:
    """Tests for company name mapping."""

    @pytest.mark.asyncio
    async def test_builds_map_from_name_and_aliases(self, mock_database: AsyncMock) -> None:
        mock_database.fetch.return_value = [
            {
                "ticker": "NVDA",
                "name": "NVIDIA Corporation",
                "aliases": ["nvidia", "geforce"],
            },
            {
                "ticker": "AMD",
                "name": "Advanced Micro Devices",
                "aliases": ["amd", "radeon"],
            },
        ]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_company_to_ticker_map()

        assert result["nvidia corporation"] == "NVDA"
        assert result["nvidia"] == "NVDA"
        assert result["geforce"] == "NVDA"
        assert result["advanced micro devices"] == "AMD"
        assert result["radeon"] == "AMD"

    @pytest.mark.asyncio
    async def test_empty_aliases(self, mock_database: AsyncMock) -> None:
        mock_database.fetch.return_value = [
            {"ticker": "X", "name": "Company X", "aliases": []},
        ]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_company_to_ticker_map()

        assert result == {"company x": "X"}


class TestSECIdentifierResolution:
    """SEC CIK and issuer-name resolution helpers."""

    @pytest.mark.asyncio
    async def test_get_by_sec_cik_normalizes_input_and_filters_active(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        mock_database.fetch.return_value = [sample_db_row]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.get_by_sec_cik("CIK1045810")

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "sec_cik = $1" in sql
        assert "is_active = TRUE" in sql
        assert args[1] == "0001045810"
        assert result[0].ticker == "NVDA"

    @pytest.mark.asyncio
    async def test_resolve_sec_identifier_orders_current_active_before_inactive_renamed(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        inactive_fb = {
            **sample_db_row,
            "ticker": "FB",
            "name": "Facebook Inc",
            "issuer_name": "Facebook Inc",
            "former_names": [],
            "sec_cik": "0001326801",
            "is_active": False,
        }
        active_meta = {
            **sample_db_row,
            "ticker": "META",
            "name": "Meta Platforms Inc",
            "issuer_name": "Meta Platforms Inc",
            "former_names": ["Facebook Inc"],
            "sec_cik": "0001326801",
            "is_active": True,
        }
        mock_database.fetch.return_value = [active_meta, inactive_fb]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.resolve_sec_identifier("Facebook Inc", active_only=False)

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "former_names" in sql
        assert "ORDER BY" in sql
        assert args[1] == "FACEBOOK INC"
        assert args[2] == "Facebook Inc"
        assert result[0].ticker == "META"
        assert result[0].is_active is True
        assert result[1].ticker == "FB"

    @pytest.mark.asyncio
    async def test_resolve_sec_identifier_handles_ambiguous_ticker_name(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        mock_database.fetch.return_value = [
            sample_db_row,
            {
                **sample_db_row,
                "ticker": "NVDA.W",
                "exchange": "US",
                "name": "NVIDIA Warrants",
                "issuer_name": "NVIDIA Corporation",
                "is_active": False,
            },
        ]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.resolve_sec_identifier("NVDA", active_only=False)

        assert result[0].ticker == "NVDA"
        assert result[0].is_active is True


class TestSearchByName:
    """Tests for fuzzy name search."""

    @pytest.mark.asyncio
    async def test_passes_threshold_and_limit(
        self, mock_database: AsyncMock, sample_db_row: dict
    ) -> None:
        sample_db_row["sim"] = 0.8  # similarity score column
        mock_database.fetch.return_value = [sample_db_row]
        repo = SecurityMasterRepository(mock_database)

        result = await repo.search_by_name("nvidia", limit=5, threshold=0.4)

        assert len(result) == 1
        args = mock_database.fetch.call_args[0]
        assert args[1] == "nvidia"
        assert args[2] == 0.4
        assert args[3] == 5


class TestDeactivate:
    """Tests for soft deactivation."""

    @pytest.mark.asyncio
    async def test_success(self, mock_database: AsyncMock) -> None:
        mock_database.execute.return_value = "UPDATE 1"
        repo = SecurityMasterRepository(mock_database)

        result = await repo.deactivate("NVDA")

        assert result is True

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        mock_database.execute.return_value = "UPDATE 0"
        repo = SecurityMasterRepository(mock_database)

        result = await repo.deactivate("AAPL")

        assert result is False


class TestCount:
    """Tests for count."""

    @pytest.mark.asyncio
    async def test_returns_count(self, mock_database: AsyncMock) -> None:
        mock_database.fetchval.return_value = 61
        repo = SecurityMasterRepository(mock_database)

        result = await repo.count()

        assert result == 61
