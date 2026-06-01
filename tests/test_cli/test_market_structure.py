"""Tests for market-structure datasource CLI commands."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_ingest_market_structure_cli_persists_local_files(
    runner: CliRunner,
    tmp_path,
) -> None:
    finra_file = tmp_path / "CNMSshvol20260601.txt"
    sec_file = tmp_path / "cnsfails202606a.txt"
    finra_file.write_text(
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260601|NVDA|600|0|1000|Q\n"
        "1\n",
    )
    sec_file.write_text(
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
        "20260601|67066G104|NVDA|12000|NVIDIA CORP|123.45\n",
    )
    mock_db = AsyncMock()
    mock_service = AsyncMock()
    mock_service.ingest_source_files.return_value = SimpleNamespace(
        total_events=2,
        finra_short_volume_count=1,
        sec_fails_to_deliver_count=1,
        upserted_count=2,
        unresolved_symbol_count=0,
    )

    with (
        patch("src.storage.database.Database", return_value=mock_db),
        patch(
            "src.market_structure.cli.MarketStructureIngestionService",
            return_value=mock_service,
        ),
    ):
        result = runner.invoke(
            main,
            [
                "ingest-market-structure",
                "--finra-short-volume-file",
                str(finra_file),
                "--sec-fails-file",
                str(sec_file),
                "--fetched-at",
                "2026-06-15T00:00:00+00:00",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_db.connect.assert_awaited_once()
    mock_db.close.assert_awaited_once()
    mock_service.ingest_source_files.assert_awaited_once()
    kwargs = mock_service.ingest_source_files.call_args.kwargs
    assert kwargs["finra_short_volume_files"][0].content.startswith("Date|Symbol")
    assert kwargs["sec_fails_to_deliver_files"][0].content.startswith("SETTLEMENT DATE")
    assert kwargs["finra_short_volume_files"][0].fetched_at == datetime(
        2026,
        6,
        15,
        tzinfo=UTC,
    )
    assert "Total events: 2" in result.output
    assert "Unresolved symbols: 0" in result.output


def test_ingest_market_structure_cli_requires_at_least_one_file(runner: CliRunner) -> None:
    result = runner.invoke(main, ["ingest-market-structure"])

    assert result.exit_code != 0
    assert "Provide at least one" in result.output
