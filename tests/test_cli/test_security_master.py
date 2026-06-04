"""Tests for security-master datasource CLI commands."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_ingest_nasdaq_trader_cli_reconciles_local_files(
    runner: CliRunner,
    tmp_path,
) -> None:
    nasdaq_file = tmp_path / "nasdaqlisted.txt"
    other_file = tmp_path / "otherlisted.txt"
    nasdaq_file.write_text(
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
        "Round Lot Size|ETF|NextShares\n"
        "NVDA|NVIDIA Corporation|Q|N|N|100|N|N\n"
        "File Creation Time: 0601202616:01|||||||\n",
    )
    other_file.write_text(
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|"
        "Test Issue|NASDAQ Symbol\n"
        "IBM|International Business Machines Corporation|N|IBM|N|100|N|IBM\n"
        "File Creation Time: 0601202616:02|||||||\n",
    )
    mock_db = AsyncMock()
    mock_service = AsyncMock()
    mock_service.ingest_nasdaq_trader_symbol_directory.return_value = SimpleNamespace(
        current_record_count=2,
        active_count=2,
        test_issue_count=0,
        deactivated_missing_count=0,
        nasdaq_listed_count=1,
        other_listed_count=1,
    )

    with (
        patch("src.storage.database.Database", return_value=mock_db),
        patch("src.security_master.service.SecurityMasterService", return_value=mock_service),
    ):
        result = runner.invoke(
            main,
            [
                "ingest-nasdaq-trader",
                "--nasdaq-listed-file",
                str(nasdaq_file),
                "--other-listed-file",
                str(other_file),
                "--observed-at",
                "2026-06-01T17:00:00+00:00",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_db.connect.assert_awaited_once()
    mock_db.close.assert_awaited_once()
    mock_service.ingest_nasdaq_trader_symbol_directory.assert_awaited_once()
    args, kwargs = mock_service.ingest_nasdaq_trader_symbol_directory.call_args
    assert "NVDA|NVIDIA Corporation" in args[0]
    assert "IBM|International Business Machines" in args[1]
    assert kwargs["observed_at"] == datetime(2026, 6, 1, 17, tzinfo=UTC)
    assert "Current records: 2" in result.output
