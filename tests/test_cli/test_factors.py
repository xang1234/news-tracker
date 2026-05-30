"""Tests for the factor datasource CLI command group."""

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main
from src.factors.refresh import FactorRefreshSummary


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _mock_db() -> AsyncMock:
    db = AsyncMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()
    return db


class TestFactorRefresh:
    def test_refresh_invokes_operational_factor_ingestion_entrypoint(
        self,
        runner: CliRunner,
    ) -> None:
        mock_db = _mock_db()
        summary = FactorRefreshSummary(
            series_seen=2,
            series_refreshed=1,
            observations_seen=3,
            observations_written=3,
            skipped_missing_credentials=["fred:DGS10"],
            errors={},
            dry_run=False,
        )

        with (
            patch("src.storage.database.Database", return_value=mock_db),
            patch(
                "src.cli.refresh_curated_factor_series",
                new=AsyncMock(return_value=summary),
            ) as refresh,
        ):
            result = runner.invoke(
                main,
                [
                    "factors",
                    "refresh",
                    "--provider",
                    "treasury",
                    "--start",
                    "2026-05-01",
                    "--end",
                    "2026-05-30",
                    "--history",
                ],
            )

        assert result.exit_code == 0, result.output
        refresh.assert_awaited_once_with(
            mock_db,
            providers={"treasury"},
            factor_ids=set(),
            start=date(2026, 5, 1),
            end=date(2026, 5, 30),
            latest=False,
            dry_run=False,
        )
        mock_db.close.assert_awaited_once()
        assert "Series refreshed: 1/2" in result.output
        assert "Skipped missing credentials: 1" in result.output
