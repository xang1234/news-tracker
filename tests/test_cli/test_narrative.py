"""Tests for the narrative CLI command group."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def _mock_db():
    db = AsyncMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.execute = AsyncMock(return_value="TRUNCATE TABLE")
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=0)
    return db


class TestNarrativeWorker:
    def test_worker_invokes_start(self, runner):
        mock_worker = AsyncMock()
        mock_worker.start = AsyncMock()
        mock_worker.stop = AsyncMock()

        with patch("src.narrative.worker.NarrativeWorker", return_value=mock_worker):
            result = runner.invoke(main, ["narrative", "worker", "--no-metrics"])

        assert result.exit_code == 0
        mock_worker.start.assert_awaited_once()


class TestNarrativeBackfill:
    def test_backfill_rejects_invalid_range(self, runner):
        mock_db = _mock_db()

        with patch("src.storage.database.Database", return_value=mock_db):
            result = runner.invoke(
                main,
                ["narrative", "backfill", "--start", "2026-02-10", "--end", "2026-02-05"],
            )

        assert result.exit_code == 0
        assert "start date must be before end date" in result.output

    def test_backfill_rebuilds_without_publishing_alerts(self, runner):
        mock_db = _mock_db()
        mock_db.fetch.return_value = [
            {
                "id": "doc_1",
                "theme_ids": ["theme_1"],
                "timestamp": datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
            }
        ]
        mock_db.fetchval.return_value = 1

        document = MagicMock(id="doc_1")
        worker = MagicMock()
        worker._doc_repo = MagicMock()
        worker._doc_repo.get_by_id = AsyncMock(return_value=document)
        worker.process_document_for_theme = AsyncMock(return_value=MagicMock(run_id="run_1"))

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.cli._build_narrative_worker_for_cli", new=AsyncMock(return_value=worker)):
            result = runner.invoke(
                main,
                ["narrative", "backfill", "--start", "2026-02-05", "--end", "2026-02-05"],
            )

        assert result.exit_code == 0
        worker.process_document_for_theme.assert_awaited_once_with(
            document,
            theme_id="theme_1",
            theme_similarity=1.0,
            publish_alerts=False,
        )
        assert "Theme assignments:    1" in result.output


class TestNarrativeReplay:
    def test_replay_dry_run_with_no_runs(self, runner):
        mock_db = _mock_db()
        mock_worker = MagicMock()
        mock_worker._narrative_repo = AsyncMock()

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.cli._build_narrative_worker_for_cli", new=AsyncMock(return_value=mock_worker)):
            result = runner.invoke(main, ["narrative", "replay", "--dry-run"])

        assert result.exit_code == 0
        assert "Runs replayed: 0" in result.output
        assert "Dry run only" in result.output


class TestNarrativeEvaluate:
    def test_evaluate_handles_empty_range(self, runner):
        mock_db = _mock_db()

        with patch("src.storage.database.Database", return_value=mock_db):
            result = runner.invoke(
                main,
                ["narrative", "evaluate", "--start", "2026-02-01", "--end", "2026-02-05"],
            )

        assert result.exit_code == 0
        assert "No narrative alerts found" in result.output
