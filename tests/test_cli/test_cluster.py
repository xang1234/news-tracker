"""Tests for the cluster CLI command group."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from src.cli import main


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def runner():
    return CliRunner()


def _make_theme(theme_id="theme_abc", doc_count=10, stage="emerging"):
    """Create a mock Theme for testing."""
    return MagicMock(
        theme_id=theme_id,
        name="test_theme",
        centroid=np.random.randn(768).astype(np.float32),
        top_keywords=["gpu", "nvidia", "architecture"],
        top_tickers=[],
        lifecycle_stage=stage,
        document_count=doc_count,
        created_at=datetime(2026, 1, 15, tzinfo=timezone.utc),
        updated_at=datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        description=None,
        top_entities=[],
        metadata={},
    )


def _make_doc(doc_id="doc_001", content="test content"):
    """Create a lightweight doc dict matching get_with_embeddings_since() output."""
    return {
        "id": doc_id,
        "content": content,
        "embedding": np.random.randn(768).astype(np.float32).tolist(),
        "authority_score": 0.5,
        "sentiment": None,
        "theme_ids": [],
    }


def _make_cluster(theme_id="theme_new"):
    """Create a mock ThemeCluster."""
    cluster = MagicMock()
    cluster.theme_id = theme_id
    cluster.name = "test_cluster"
    cluster.topic_words = [("gpu", 0.9), ("nvidia", 0.8), ("chip", 0.7)]
    cluster.centroid = np.random.randn(768).astype(np.float32)
    cluster.document_count = 5
    cluster.document_ids = ["d1", "d2"]
    cluster.metadata = {"lifecycle_stage": "emerging"}
    return cluster


def _mock_db():
    """Create a mock Database that works as async context."""
    db = AsyncMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.execute = AsyncMock(return_value="UPDATE 1")
    db.fetch = AsyncMock(return_value=[])
    return db


# ── cluster fit ──────────────────────────────────────────


class TestClusterFit:
    """Tests for `cluster fit` command."""

    def test_fit_no_documents(self, runner):
        mock_db = _mock_db()
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_with_embeddings_since = AsyncMock(return_value=[])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.storage.repository.DocumentRepository", return_value=mock_doc_repo), \
             patch("src.themes.repository.ThemeRepository"):
            result = runner.invoke(main, ["cluster", "fit", "--days", "7"])

        assert result.exit_code == 0
        assert "No documents with embeddings found" in result.output

    def test_fit_discovers_themes(self, runner):
        mock_db = _mock_db()
        docs = [_make_doc(f"doc_{i}") for i in range(5)]

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)
        mock_theme_repo = AsyncMock()

        cluster = _make_cluster("theme_new_123")
        mock_service = MagicMock()
        mock_service.fit.return_value = {"theme_new_123": cluster}

        mock_theme_obj = MagicMock()

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.storage.repository.DocumentRepository", return_value=mock_doc_repo), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo), \
             patch("src.clustering.service.BERTopicService", return_value=mock_service), \
             patch("src.clustering.daily_job._cluster_to_theme", return_value=mock_theme_obj):
            result = runner.invoke(main, ["cluster", "fit"])

        assert result.exit_code == 0
        assert "Documents processed: 5" in result.output
        assert "Themes discovered:   1" in result.output
        assert "Themes persisted:    1" in result.output
        mock_theme_repo.create.assert_called_once_with(mock_theme_obj)

    def test_fit_no_themes_discovered(self, runner):
        mock_db = _mock_db()
        docs = [_make_doc(f"doc_{i}") for i in range(3)]

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)

        mock_service = MagicMock()
        mock_service.fit.return_value = {}

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.storage.repository.DocumentRepository", return_value=mock_doc_repo), \
             patch("src.themes.repository.ThemeRepository"), \
             patch("src.clustering.service.BERTopicService", return_value=mock_service):
            result = runner.invoke(main, ["cluster", "fit"])

        assert result.exit_code == 0
        assert "No themes discovered" in result.output

    def test_fit_handles_create_error(self, runner):
        mock_db = _mock_db()
        docs = [_make_doc(f"doc_{i}") for i in range(3)]

        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)
        mock_theme_repo = AsyncMock()
        mock_theme_repo.create = AsyncMock(side_effect=Exception("duplicate key"))

        cluster = _make_cluster("theme_err")
        mock_service = MagicMock()
        mock_service.fit.return_value = {"theme_err": cluster}

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.storage.repository.DocumentRepository", return_value=mock_doc_repo), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo), \
             patch("src.clustering.service.BERTopicService", return_value=mock_service), \
             patch("src.clustering.daily_job._cluster_to_theme"):
            result = runner.invoke(main, ["cluster", "fit"])

        assert result.exit_code == 0
        assert "Failed to create" in result.output
        assert "Themes persisted:    0" in result.output


# ── cluster run ──────────────────────────────────────────


class TestClusterRun:
    """Tests for `cluster run` command."""

    def test_run_dry_run(self, runner):
        mock_db = _mock_db()
        mock_doc_repo = AsyncMock()
        mock_doc_repo.get_with_embeddings_since = AsyncMock(return_value=[_make_doc()])
        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=[_make_theme()])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.storage.repository.DocumentRepository", return_value=mock_doc_repo), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "run", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "Documents with embeddings: 1" in result.output
        assert "Existing themes: 1" in result.output

    def test_run_executes(self, runner):
        mock_db = _mock_db()

        mock_result = MagicMock()
        mock_result.date = date(2026, 2, 6)
        mock_result.documents_fetched = 50
        mock_result.documents_assigned = 45
        mock_result.documents_unassigned = 5
        mock_result.new_themes_created = 1
        mock_result.themes_merged = 0
        mock_result.metrics_computed = 3
        mock_result.errors = []
        mock_result.elapsed_seconds = 1.23

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.clustering.daily_job.run_daily_clustering",
                   new_callable=AsyncMock, return_value=mock_result):
            result = runner.invoke(main, ["cluster", "run", "--date", "2026-02-06"])

        assert result.exit_code == 0
        assert "Documents fetched:  50" in result.output
        assert "Documents assigned: 45" in result.output

    def test_run_shows_errors(self, runner):
        mock_db = _mock_db()

        mock_result = MagicMock()
        mock_result.date = date(2026, 2, 6)
        mock_result.documents_fetched = 10
        mock_result.documents_assigned = 8
        mock_result.documents_unassigned = 2
        mock_result.new_themes_created = 0
        mock_result.themes_merged = 0
        mock_result.metrics_computed = 0
        mock_result.errors = ["fetch_docs: timeout"]
        mock_result.elapsed_seconds = 5.0

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.clustering.daily_job.run_daily_clustering",
                   new_callable=AsyncMock, return_value=mock_result):
            result = runner.invoke(main, ["cluster", "run"])

        assert result.exit_code == 0
        assert "Errors:" in result.output
        assert "fetch_docs: timeout" in result.output


# ── cluster backfill ─────────────────────────────────────


class TestClusterBackfill:
    """Tests for `cluster backfill` command."""

    def test_backfill_date_range(self, runner):
        mock_db = _mock_db()

        mock_result = MagicMock()
        mock_result.documents_fetched = 10
        mock_result.documents_assigned = 8
        mock_result.errors = []

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.clustering.daily_job.run_daily_clustering",
                   new_callable=AsyncMock, return_value=mock_result):
            result = runner.invoke(main, [
                "cluster", "backfill",
                "--start", "2026-02-01", "--end", "2026-02-03",
            ])

        assert result.exit_code == 0
        assert "Backfilling 3 days" in result.output
        assert "2026-02-01" in result.output
        assert "2026-02-02" in result.output
        assert "2026-02-03" in result.output
        assert "3 succeeded, 0 failed" in result.output

    def test_backfill_start_after_end(self, runner):
        mock_db = _mock_db()

        with patch("src.storage.database.Database", return_value=mock_db):
            result = runner.invoke(main, [
                "cluster", "backfill",
                "--start", "2026-02-05", "--end", "2026-02-01",
            ])

        assert result.exit_code == 0
        assert "start date must be before end date" in result.output

    def test_backfill_continues_on_error(self, runner):
        mock_db = _mock_db()

        mock_result = MagicMock()
        mock_result.documents_fetched = 5
        mock_result.documents_assigned = 5
        mock_result.errors = []

        call_count = 0

        async def side_effect(db, target_date=None, config=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Some processing error")
            return mock_result

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.clustering.daily_job.run_daily_clustering",
                   side_effect=side_effect):
            result = runner.invoke(main, [
                "cluster", "backfill",
                "--start", "2026-02-01", "--end", "2026-02-03",
            ])

        assert result.exit_code == 0
        assert "2 succeeded, 1 failed" in result.output
        assert "FAIL" in result.output

    def test_backfill_stops_on_connection_error(self, runner):
        mock_db = _mock_db()

        mock_result = MagicMock()
        mock_result.documents_fetched = 5
        mock_result.documents_assigned = 5
        mock_result.errors = []

        call_count = 0

        async def side_effect(db, target_date=None, config=None):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("connection pool exhausted")
            return mock_result

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.clustering.daily_job.run_daily_clustering",
                   side_effect=side_effect):
            result = runner.invoke(main, [
                "cluster", "backfill",
                "--start", "2026-02-01", "--end", "2026-02-05",
            ])

        assert result.exit_code == 0
        assert "Fatal: DB connection lost" in result.output
        assert "1 succeeded, 1 failed" in result.output


# ── cluster merge ────────────────────────────────────────


class TestClusterMerge:
    """Tests for `cluster merge` command."""

    def test_merge_too_few_themes(self, runner):
        mock_db = _mock_db()
        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=[_make_theme()])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "merge"])

        assert result.exit_code == 0
        assert "Only 1 theme(s) found" in result.output

    def test_merge_dry_run(self, runner):
        mock_db = _mock_db()
        themes = [_make_theme("theme_a"), _make_theme("theme_b")]

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=themes)

        mock_service = MagicMock()
        mock_service.merge_similar_themes.return_value = [("theme_a", "theme_b")]

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo), \
             patch("src.clustering.service.BERTopicService", return_value=mock_service), \
             patch("src.clustering.daily_job._theme_to_cluster"):
            result = runner.invoke(main, ["cluster", "merge", "--dry-run"])

        assert result.exit_code == 0
        assert "1 merge(s) would occur" in result.output

    def test_merge_dry_run_nothing_to_merge(self, runner):
        mock_db = _mock_db()
        themes = [_make_theme("theme_a"), _make_theme("theme_b")]

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=themes)

        mock_service = MagicMock()
        mock_service.merge_similar_themes.return_value = []

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo), \
             patch("src.clustering.service.BERTopicService", return_value=mock_service), \
             patch("src.clustering.daily_job._theme_to_cluster"):
            result = runner.invoke(main, ["cluster", "merge", "--dry-run"])

        assert result.exit_code == 0
        assert "No themes similar enough to merge" in result.output

    def test_merge_executes(self, runner):
        mock_db = _mock_db()
        themes = [_make_theme("theme_a"), _make_theme("theme_b")]

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=themes)

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo), \
             patch("src.clustering.daily_job._run_weekly_merge",
                   new_callable=AsyncMock, return_value=1):
            result = runner.invoke(main, ["cluster", "merge"])

        assert result.exit_code == 0
        assert "Merged 1 theme(s) successfully" in result.output

    def test_merge_custom_threshold(self, runner):
        mock_db = _mock_db()
        themes = [_make_theme("theme_a"), _make_theme("theme_b")]

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=themes)

        mock_service = MagicMock()
        mock_service.merge_similar_themes.return_value = [("theme_a", "theme_b")]

        captured_config = {}

        def capture_config(*args, **kwargs):
            config = MagicMock()
            captured_config["instance"] = config
            return config

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo), \
             patch("src.clustering.service.BERTopicService", return_value=mock_service), \
             patch("src.clustering.daily_job._theme_to_cluster"), \
             patch("src.clustering.config.ClusteringConfig", side_effect=capture_config):
            result = runner.invoke(main, [
                "cluster", "merge", "--dry-run", "--threshold", "0.80",
            ])

        assert result.exit_code == 0
        assert captured_config["instance"].similarity_threshold_merge == 0.80


# ── cluster status ───────────────────────────────────────


class TestClusterStatus:
    """Tests for `cluster status` command."""

    def test_status_no_themes(self, runner):
        mock_db = _mock_db()
        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=[])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "status"])

        assert result.exit_code == 0
        assert "No themes found" in result.output

    def test_status_shows_summary(self, runner):
        mock_db = _mock_db()
        themes = [
            _make_theme("t1", doc_count=100, stage="emerging"),
            _make_theme("t2", doc_count=200, stage="accelerating"),
            _make_theme("t3", doc_count=50, stage="emerging"),
            _make_theme("t4", doc_count=300, stage="mature"),
        ]

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=themes)

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "status"])

        assert result.exit_code == 0
        assert "Total themes:     4" in result.output
        assert "Total documents:  650" in result.output
        assert "emerging" in result.output
        assert "accelerating" in result.output
        assert "mature" in result.output


# ── cluster recompute-centroids ──────────────────────────


class TestClusterRecomputeCentroids:
    """Tests for `cluster recompute-centroids` command."""

    def test_recompute_no_themes(self, runner):
        mock_db = _mock_db()
        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=[])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "recompute-centroids"])

        assert result.exit_code == 0
        assert "No themes found" in result.output

    def test_recompute_updates_centroids(self, runner):
        mock_db = _mock_db()
        theme = _make_theme("theme_xyz", doc_count=5)

        emb = np.random.randn(768).astype(np.float32).tolist()
        mock_db.fetch = AsyncMock(return_value=[
            {"embedding": emb},
            {"embedding": emb},
        ])

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=[theme])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "recompute-centroids"])

        assert result.exit_code == 0
        assert "updated (2 docs)" in result.output
        assert "1 updated, 0 skipped" in result.output
        mock_theme_repo.update_centroid.assert_called_once()

    def test_recompute_skips_empty_themes(self, runner):
        mock_db = _mock_db()
        theme = _make_theme("theme_empty")

        mock_db.fetch = AsyncMock(return_value=[])

        mock_theme_repo = AsyncMock()
        mock_theme_repo.get_all = AsyncMock(return_value=[theme])

        with patch("src.storage.database.Database", return_value=mock_db), \
             patch("src.themes.repository.ThemeRepository", return_value=mock_theme_repo):
            result = runner.invoke(main, ["cluster", "recompute-centroids"])

        assert result.exit_code == 0
        assert "skipped (no embeddings)" in result.output
        assert "0 updated, 1 skipped" in result.output


# ── cluster help ─────────────────────────────────────────


class TestClusterHelp:
    """Tests for `cluster --help` and subcommand help."""

    def test_cluster_group_help(self, runner):
        result = runner.invoke(main, ["cluster", "--help"])
        assert result.exit_code == 0
        assert "fit" in result.output
        assert "run" in result.output
        assert "backfill" in result.output
        assert "merge" in result.output
        assert "status" in result.output
        assert "recompute-centroids" in result.output

    def test_fit_help(self, runner):
        result = runner.invoke(main, ["cluster", "fit", "--help"])
        assert result.exit_code == 0
        assert "--days" in result.output

    def test_backfill_help(self, runner):
        result = runner.invoke(main, ["cluster", "backfill", "--help"])
        assert result.exit_code == 0
        assert "--start" in result.output
        assert "--end" in result.output

    def test_merge_help(self, runner):
        result = runner.invoke(main, ["cluster", "merge", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "--threshold" in result.output
