"""Tests for BacktestRun and BacktestRunRepository lifecycle."""

import json
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.backtest.audit import BacktestRun, BacktestRunRepository


class TestBacktestRunRepository:
    """Test BacktestRunRepository CRUD operations."""

    @pytest.mark.asyncio
    async def test_create(
        self,
        mock_database: AsyncMock,
        sample_backtest_run: BacktestRun,
        sample_backtest_run_row: dict,
    ) -> None:
        """create() inserts a new run record."""
        mock_database.fetchrow.return_value = sample_backtest_run_row
        repo = BacktestRunRepository(mock_database)

        result = await repo.create(sample_backtest_run)

        sql = mock_database.fetchrow.call_args[0][0]
        assert "INSERT INTO backtest_runs" in sql
        assert result.run_id == "run_test_001"
        assert result.status == "running"

    @pytest.mark.asyncio
    async def test_create_params(
        self,
        mock_database: AsyncMock,
        sample_backtest_run: BacktestRun,
        sample_backtest_run_row: dict,
    ) -> None:
        """create() serializes parameters to JSON."""
        mock_database.fetchrow.return_value = sample_backtest_run_row
        repo = BacktestRunRepository(mock_database)

        await repo.create(sample_backtest_run)

        args = mock_database.fetchrow.call_args[0]
        assert args[1] == sample_backtest_run.run_id
        assert args[2] == sample_backtest_run.model_version_id
        # parameters should be JSON string
        assert json.loads(args[5]) == sample_backtest_run.parameters

    @pytest.mark.asyncio
    async def test_mark_completed(
        self,
        mock_database: AsyncMock,
        sample_backtest_run_row: dict,
    ) -> None:
        """mark_completed() transitions status and stores results."""
        completed_row = {
            **sample_backtest_run_row,
            "status": "completed",
            "results": '{"accuracy": 0.85}',
            "completed_at": datetime(2025, 7, 1, 11, 0, 0, tzinfo=timezone.utc),
        }
        mock_database.fetchrow.return_value = completed_row
        repo = BacktestRunRepository(mock_database)

        result = await repo.mark_completed(
            "run_test_001", {"accuracy": 0.85},
        )

        sql = mock_database.fetchrow.call_args[0][0]
        assert "status = 'completed'" in sql
        assert result.status == "completed"
        assert result.results == {"accuracy": 0.85}

    @pytest.mark.asyncio
    async def test_mark_completed_not_found(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """mark_completed() raises if run not found."""
        mock_database.fetchrow.return_value = None
        repo = BacktestRunRepository(mock_database)

        with pytest.raises(ValueError, match="not found"):
            await repo.mark_completed("run_missing", {})

    @pytest.mark.asyncio
    async def test_mark_failed(
        self,
        mock_database: AsyncMock,
        sample_backtest_run_row: dict,
    ) -> None:
        """mark_failed() transitions status and stores error."""
        failed_row = {
            **sample_backtest_run_row,
            "status": "failed",
            "error_message": "OOM",
            "completed_at": datetime(2025, 7, 1, 11, 0, 0, tzinfo=timezone.utc),
        }
        mock_database.fetchrow.return_value = failed_row
        repo = BacktestRunRepository(mock_database)

        result = await repo.mark_failed("run_test_001", "OOM")

        sql = mock_database.fetchrow.call_args[0][0]
        assert "status = 'failed'" in sql
        assert result.status == "failed"
        assert result.error_message == "OOM"

    @pytest.mark.asyncio
    async def test_mark_failed_not_found(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """mark_failed() raises if run not found."""
        mock_database.fetchrow.return_value = None
        repo = BacktestRunRepository(mock_database)

        with pytest.raises(ValueError, match="not found"):
            await repo.mark_failed("run_missing", "error")

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self,
        mock_database: AsyncMock,
        sample_backtest_run_row: dict,
    ) -> None:
        mock_database.fetchrow.return_value = sample_backtest_run_row
        repo = BacktestRunRepository(mock_database)

        result = await repo.get_by_id("run_test_001")

        assert result is not None
        assert result.run_id == "run_test_001"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self,
        mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchrow.return_value = None
        repo = BacktestRunRepository(mock_database)

        result = await repo.get_by_id("run_missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_runs_no_filter(
        self,
        mock_database: AsyncMock,
        sample_backtest_run_row: dict,
    ) -> None:
        """list_runs() without status filter returns all runs."""
        mock_database.fetch.return_value = [sample_backtest_run_row]
        repo = BacktestRunRepository(mock_database)

        results = await repo.list_runs()

        sql = mock_database.fetch.call_args[0][0]
        assert "WHERE" not in sql
        assert "ORDER BY created_at DESC" in sql
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_runs_with_status(
        self,
        mock_database: AsyncMock,
        sample_backtest_run_row: dict,
    ) -> None:
        """list_runs() with status filter uses WHERE clause."""
        mock_database.fetch.return_value = [sample_backtest_run_row]
        repo = BacktestRunRepository(mock_database)

        results = await repo.list_runs(status="running")

        sql = mock_database.fetch.call_args[0][0]
        assert "WHERE status = $1" in sql
        args = mock_database.fetch.call_args[0]
        assert args[1] == "running"

    @pytest.mark.asyncio
    async def test_jsonb_parsing(
        self,
        mock_database: AsyncMock,
        sample_backtest_run_row: dict,
    ) -> None:
        """JSONB string fields are parsed to Python dicts."""
        mock_database.fetchrow.return_value = sample_backtest_run_row
        repo = BacktestRunRepository(mock_database)

        result = await repo.get_by_id("run_test_001")

        assert isinstance(result.parameters, dict)
        assert result.parameters["strategy"] == "swing"
