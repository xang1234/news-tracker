"""Tests for the graph CLI command group."""

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from src.cli import main
from src.graph.seed_data import ALL_EDGES, ALL_NODES, SEED_VERSION


@pytest.fixture
def runner():
    return CliRunner()


class TestGraphSeed:
    """Test the `graph seed` CLI command."""

    def test_seed_success(self, runner: CliRunner) -> None:
        """graph seed prints summary on success."""
        mock_db = AsyncMock()
        mock_db.fetchrow.return_value = {
            "node_id": "X",
            "node_type": "ticker",
            "name": "X",
            "metadata": "{}",
            "created_at": None,
            "updated_at": None,
        }

        with patch("src.graph.seed_data.Database", return_value=mock_db) as MockDB:
            # Also patch the Database import inside the CLI function
            with patch("src.storage.database.Database", return_value=mock_db):
                result = runner.invoke(main, ["graph", "seed"])

        assert result.exit_code == 0, result.output
        assert "Graph Seed Results" in result.output
        assert f"Nodes seeded: {len(ALL_NODES)}" in result.output
        assert f"Edges seeded: {len(ALL_EDGES)}" in result.output
        assert "successfully" in result.output

    def test_seed_closes_db_on_error(self, runner: CliRunner) -> None:
        """graph seed closes DB connection even on failure."""
        mock_db = AsyncMock()
        mock_db.fetchrow.side_effect = RuntimeError("connection lost")

        with patch("src.storage.database.Database", return_value=mock_db):
            result = runner.invoke(main, ["graph", "seed"])

        # DB should be closed despite the error
        mock_db.close.assert_called_once()
        assert result.exit_code != 0
