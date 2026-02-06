"""Tests for daily batch clustering job."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.daily_job import (
    DailyClusteringResult,
    _aggregate_sentiment_metrics,
    _batch_cosine_similarity,
    _cluster_to_theme,
    _theme_to_cluster,
    run_daily_clustering,
)
from src.clustering.schemas import ThemeCluster
from src.themes.schemas import Theme, ThemeMetrics


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def config():
    return ClusteringConfig(
        similarity_threshold_assign=0.75,
        similarity_threshold_new=0.30,
        centroid_learning_rate=0.01,
        hdbscan_min_cluster_size=10,
    )


@pytest.fixture
def mock_database():
    db = AsyncMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.execute = AsyncMock(return_value="UPDATE 1")
    return db


@pytest.fixture
def mock_doc_repo():
    return AsyncMock()


@pytest.fixture
def mock_theme_repo():
    return AsyncMock()


def _make_theme(theme_id="theme_abc", centroid=None, doc_count=10, keywords=None):
    """Create a Theme for testing."""
    c = centroid if centroid is not None else np.random.randn(768).astype(np.float32)
    return Theme(
        theme_id=theme_id,
        name="test_theme",
        centroid=c,
        top_keywords=keywords or ["gpu", "nvidia", "architecture"],
        document_count=doc_count,
        lifecycle_stage="emerging",
    )


def _make_doc(doc_id="doc_001", embedding=None, content="test content",
              authority=0.5, sentiment=None, theme_ids=None):
    """Create a lightweight doc dict matching get_with_embeddings_since() output."""
    emb = embedding if embedding is not None else np.random.randn(768).astype(np.float32).tolist()
    return {
        "id": doc_id,
        "content": content,
        "embedding": emb,
        "authority_score": authority,
        "sentiment": sentiment,
        "theme_ids": theme_ids or [],
    }


# ── TestBatchCosineSimilarity ──────────────────────────────


class TestBatchCosineSimilarity:
    """Tests for the _batch_cosine_similarity helper."""

    def test_identical_embeddings_similarity_one(self):
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        centroids = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        result = _batch_cosine_similarity(emb, centroids)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-6)

    def test_orthogonal_embeddings_similarity_zero(self):
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        centroids = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        result = _batch_cosine_similarity(emb, centroids)
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-6)

    def test_opposite_embeddings_similarity_negative(self):
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        centroids = np.array([[-1.0, 0.0, 0.0]], dtype=np.float32)
        result = _batch_cosine_similarity(emb, centroids)
        np.testing.assert_allclose(result[0, 0], -1.0, atol=1e-6)

    def test_output_shape(self):
        n_docs, n_themes, dim = 50, 5, 768
        emb = np.random.randn(n_docs, dim).astype(np.float32)
        centroids = np.random.randn(n_themes, dim).astype(np.float32)
        result = _batch_cosine_similarity(emb, centroids)
        assert result.shape == (n_docs, n_themes)

    def test_zero_norm_handling(self):
        """Zero-norm embedding shouldn't produce NaN (guarded by np.where)."""
        emb = np.zeros((1, 768), dtype=np.float32)
        centroids = np.ones((1, 768), dtype=np.float32)
        result = _batch_cosine_similarity(emb, centroids)
        assert not np.any(np.isnan(result))

    def test_values_in_range(self):
        """All similarity values should be in [-1, 1]."""
        emb = np.random.randn(20, 768).astype(np.float32)
        centroids = np.random.randn(5, 768).astype(np.float32)
        result = _batch_cosine_similarity(emb, centroids)
        assert np.all(result >= -1.0 - 1e-6)
        assert np.all(result <= 1.0 + 1e-6)


# ── TestThemeConversion ───────────────────────────────────


class TestThemeConversion:
    """Tests for _theme_to_cluster and _cluster_to_theme round-trip."""

    def test_theme_to_cluster(self):
        theme = _make_theme(keywords=["gpu", "nvidia", "architecture"])
        cluster = _theme_to_cluster(theme)
        assert cluster.theme_id == theme.theme_id
        assert cluster.name == theme.name
        assert len(cluster.topic_words) == 3
        assert cluster.topic_words[0] == ("gpu", 1.0)
        assert cluster.topic_words[1][0] == "nvidia"
        np.testing.assert_array_equal(cluster.centroid, theme.centroid)

    def test_cluster_to_theme(self):
        cluster = ThemeCluster(
            theme_id="theme_xyz",
            name="test_cluster",
            topic_words=[("chip", 0.9), ("design", 0.8)],
            centroid=np.zeros(768, dtype=np.float32),
            document_count=42,
            metadata={"lifecycle_stage": "accelerating"},
        )
        theme = _cluster_to_theme(cluster)
        assert theme.theme_id == "theme_xyz"
        assert theme.top_keywords == ["chip", "design"]
        assert theme.lifecycle_stage == "accelerating"
        assert theme.document_count == 42

    def test_cluster_to_theme_default_lifecycle(self):
        cluster = ThemeCluster(
            theme_id="theme_default",
            name="default",
            topic_words=[("test", 1.0)],
            centroid=np.zeros(768, dtype=np.float32),
            document_count=1,
            metadata={},
        )
        theme = _cluster_to_theme(cluster)
        assert theme.lifecycle_stage == "emerging"


# ── TestDailyClusteringAssignment ─────────────────────────


class TestDailyClusteringAssignment:

    @pytest.mark.asyncio
    async def test_strong_assignment_above_threshold(self, config, mock_database):
        """Documents above assign threshold get strong assignment + EMA."""
        # Create a theme centroid and a very similar document embedding
        centroid = np.ones(768, dtype=np.float32)
        centroid /= np.linalg.norm(centroid)
        embedding = centroid + np.random.randn(768).astype(np.float32) * 0.01
        embedding /= np.linalg.norm(embedding)

        theme = _make_theme(centroid=centroid)
        doc = _make_doc(embedding=embedding.tolist())

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[doc])
            doc_repo.update_themes = AsyncMock(return_value=True)
            doc_repo.get_sentiments_for_theme = AsyncMock(return_value=[])
            theme_repo.get_all = AsyncMock(return_value=[theme])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()

            result = await run_daily_clustering(mock_database, config=config)

        assert result.documents_fetched == 1
        assert result.documents_assigned == 1
        assert result.documents_unassigned == 0
        doc_repo.update_themes.assert_called_once()

    @pytest.mark.asyncio
    async def test_weak_assignment_between_thresholds(self, config, mock_database):
        """Documents between new_threshold and assign_threshold get weak assignment."""
        # Create embeddings with moderate similarity (~0.5)
        centroid = np.zeros(768, dtype=np.float32)
        centroid[0] = 1.0

        # Embedding with ~60% similarity (between 0.30 and 0.75)
        embedding = np.zeros(768, dtype=np.float32)
        embedding[0] = 0.6
        embedding[1] = 0.8
        embedding /= np.linalg.norm(embedding)

        theme = _make_theme(centroid=centroid)
        doc = _make_doc(embedding=embedding.tolist())

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[doc])
            doc_repo.update_themes = AsyncMock(return_value=True)
            doc_repo.get_sentiments_for_theme = AsyncMock(return_value=[])
            theme_repo.get_all = AsyncMock(return_value=[theme])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()

            result = await run_daily_clustering(mock_database, config=config)

        assert result.documents_assigned == 1
        # Weak assignment: update_themes called but no centroid update
        doc_repo.update_themes.assert_called_once()

    @pytest.mark.asyncio
    async def test_unassigned_below_new_threshold(self, config, mock_database):
        """Documents with very low similarity go unassigned."""
        # Orthogonal embeddings: similarity ~0
        centroid = np.zeros(768, dtype=np.float32)
        centroid[0] = 1.0

        embedding = np.zeros(768, dtype=np.float32)
        embedding[1] = 1.0  # orthogonal

        theme = _make_theme(centroid=centroid)
        doc = _make_doc(embedding=embedding.tolist())

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[doc])
            doc_repo.update_themes = AsyncMock(return_value=True)
            theme_repo.get_all = AsyncMock(return_value=[theme])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()

            result = await run_daily_clustering(mock_database, config=config)

        assert result.documents_unassigned == 1
        assert result.documents_assigned == 0
        doc_repo.update_themes.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_documents_returns_zeros(self, config, mock_database):
        """No documents in the time window returns zero counts."""
        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository"):
            doc_repo = MockDocRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[])

            result = await run_daily_clustering(mock_database, config=config)

        assert result.documents_fetched == 0
        assert result.documents_assigned == 0
        assert result.documents_unassigned == 0

    @pytest.mark.asyncio
    async def test_no_existing_themes_all_unassigned(self, config, mock_database):
        """No themes in DB means all docs are unassigned."""
        docs = [_make_doc(doc_id=f"doc_{i}") for i in range(5)]

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)
            theme_repo.get_all = AsyncMock(return_value=[])

            result = await run_daily_clustering(mock_database, config=config)

        assert result.documents_fetched == 5
        assert result.documents_unassigned == 5
        assert result.documents_assigned == 0


# ── TestNewThemeDetection ─────────────────────────────────


class TestNewThemeDetection:

    @pytest.mark.asyncio
    async def test_new_themes_from_unassigned(self, config, mock_database):
        """Unassigned candidates should trigger check_new_themes."""
        # Create orthogonal docs that won't match the existing theme
        centroid = np.zeros(768, dtype=np.float32)
        centroid[0] = 1.0
        theme = _make_theme(centroid=centroid)

        # Create enough unassigned docs (need min_candidates)
        docs = []
        for i in range(10):
            emb = np.zeros(768, dtype=np.float32)
            emb[i + 1] = 1.0  # orthogonal to centroid[0]
            docs.append(_make_doc(doc_id=f"doc_{i}", embedding=emb.tolist(),
                                  content=f"unique content {i}"))

        new_cluster = ThemeCluster(
            theme_id="theme_new_123",
            name="new_theme",
            topic_words=[("emerging", 0.9)],
            centroid=np.random.randn(768).astype(np.float32),
            document_count=5,
            metadata={"lifecycle_stage": "emerging"},
        )

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo, \
             patch("src.clustering.daily_job.BERTopicService") as MockService:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value
            service_instance = MockService.return_value

            doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)
            doc_repo.update_themes = AsyncMock(return_value=True)
            doc_repo.get_sentiments_for_theme = AsyncMock(return_value=[])
            theme_repo.get_all = AsyncMock(return_value=[theme])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()
            theme_repo.create = AsyncMock(return_value=_make_theme())

            service_instance.check_new_themes.return_value = [new_cluster]

            result = await run_daily_clustering(mock_database, config=config)

        assert result.new_themes_created == 1
        theme_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_too_few_candidates_skips_detection(self, config, mock_database):
        """Fewer candidates than min_candidates skips new theme detection."""
        centroid = np.zeros(768, dtype=np.float32)
        centroid[0] = 1.0
        theme = _make_theme(centroid=centroid)

        # Only 2 unassigned docs — below threshold
        docs = []
        for i in range(2):
            emb = np.zeros(768, dtype=np.float32)
            emb[i + 1] = 1.0
            docs.append(_make_doc(doc_id=f"doc_{i}", embedding=emb.tolist()))

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)
            theme_repo.get_all = AsyncMock(return_value=[theme])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()

            result = await run_daily_clustering(mock_database, config=config)

        assert result.new_themes_created == 0


# ── TestMetricsComputation ────────────────────────────────


class TestMetricsComputation:
    """Tests for _aggregate_sentiment_metrics helper."""

    def test_basic_metrics(self):
        sentiments = [
            {"sentiment": {"label": "positive", "scores": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}},
             "authority_score": 0.8},
            {"sentiment": {"label": "negative", "scores": {"positive": 0.1, "negative": 0.8, "neutral": 0.1}},
             "authority_score": 0.6},
        ]
        m = _aggregate_sentiment_metrics("theme_1", date(2026, 2, 5), 2, sentiments)
        assert m.document_count == 2
        assert m.sentiment_score is not None
        assert m.avg_authority == pytest.approx(0.7)
        assert m.bullish_ratio == pytest.approx(0.5)

    def test_all_positive_sentiment(self):
        sentiments = [
            {"sentiment": {"label": "positive", "scores": {"positive": 0.9, "negative": 0.05, "neutral": 0.05}},
             "authority_score": 0.9},
        ]
        m = _aggregate_sentiment_metrics("theme_1", date(2026, 2, 5), 1, sentiments)
        assert m.bullish_ratio == pytest.approx(1.0)
        # pos - neg = 0.85
        assert m.sentiment_score == pytest.approx(0.85)

    def test_no_sentiment_returns_none(self):
        m = _aggregate_sentiment_metrics("theme_1", date(2026, 2, 5), 3, [])
        assert m.sentiment_score is None
        assert m.avg_authority is None
        assert m.bullish_ratio is None
        assert m.document_count == 3

    def test_missing_authority_excluded(self):
        sentiments = [
            {"sentiment": {"label": "neutral", "scores": {"positive": 0.3, "negative": 0.3, "neutral": 0.4}},
             "authority_score": None},
        ]
        m = _aggregate_sentiment_metrics("theme_1", date(2026, 2, 5), 1, sentiments)
        assert m.avg_authority is None  # No valid authority values
        assert m.sentiment_score == pytest.approx(0.0)  # pos - neg = 0

    def test_volume_zscore_velocity_acceleration_none(self):
        """Feature 3.1 fields should always be None."""
        m = _aggregate_sentiment_metrics("theme_1", date(2026, 2, 5), 1, [])
        assert m.volume_zscore is None
        assert m.velocity is None
        assert m.acceleration is None


# ── TestWeeklyMaintenance ─────────────────────────────────


class TestWeeklyMaintenance:

    @pytest.mark.asyncio
    async def test_merge_triggered_on_monday(self, config, mock_database):
        """Monday target_date should trigger weekly merge."""
        # 2026-02-02 is a Monday
        monday = date(2026, 2, 2)

        centroid_a = np.ones(768, dtype=np.float32) / np.sqrt(768)
        centroid_b = centroid_a + np.random.randn(768).astype(np.float32) * 0.001
        centroid_b /= np.linalg.norm(centroid_b)

        theme_a = _make_theme(theme_id="theme_a", centroid=centroid_a, doc_count=50,
                              keywords=["gpu", "nvidia"])
        theme_b = _make_theme(theme_id="theme_b", centroid=centroid_b, doc_count=30,
                              keywords=["gpu", "nvidia"])

        doc = _make_doc(embedding=centroid_a.tolist())

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value

            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[doc])
            doc_repo.update_themes = AsyncMock(return_value=True)
            doc_repo.get_sentiments_for_theme = AsyncMock(return_value=[])
            theme_repo.get_all = AsyncMock(return_value=[theme_a, theme_b])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()
            theme_repo.delete = AsyncMock(return_value=True)
            theme_repo.update = AsyncMock(return_value=theme_a)

            result = await run_daily_clustering(
                mock_database, target_date=monday, config=config,
            )

        # At least one merge should happen (themes are nearly identical)
        assert result.themes_merged >= 1
        theme_repo.delete.assert_called()

    @pytest.mark.asyncio
    async def test_no_merge_on_other_days(self, config, mock_database):
        """Non-Monday days should not trigger merge."""
        # 2026-02-03 is a Tuesday
        tuesday = date(2026, 2, 3)

        centroid = np.ones(768, dtype=np.float32) / np.sqrt(768)
        theme = _make_theme(centroid=centroid)
        doc = _make_doc(embedding=centroid.tolist())

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value

            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[doc])
            doc_repo.update_themes = AsyncMock(return_value=True)
            doc_repo.get_sentiments_for_theme = AsyncMock(return_value=[])
            theme_repo.get_all = AsyncMock(return_value=[theme])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()

            result = await run_daily_clustering(
                mock_database, target_date=tuesday, config=config,
            )

        assert result.themes_merged == 0


# ── TestRunDailyClustering (integration-style) ──────────────


class TestRunDailyClustering:

    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self, config, mock_database):
        """End-to-end test with multiple docs and themes."""
        # Create two themes
        centroid_a = np.zeros(768, dtype=np.float32)
        centroid_a[0] = 1.0
        centroid_b = np.zeros(768, dtype=np.float32)
        centroid_b[1] = 1.0

        theme_a = _make_theme(theme_id="theme_a", centroid=centroid_a,
                              keywords=["topic_a"])
        theme_b = _make_theme(theme_id="theme_b", centroid=centroid_b,
                              keywords=["topic_b"])

        # Docs close to theme_a
        docs = []
        for i in range(3):
            emb = centroid_a + np.random.randn(768).astype(np.float32) * 0.01
            emb /= np.linalg.norm(emb)
            docs.append(_make_doc(
                doc_id=f"doc_a_{i}", embedding=emb.tolist(),
                sentiment={"label": "positive", "scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1}},
            ))

        # Doc close to theme_b
        emb_b = centroid_b + np.random.randn(768).astype(np.float32) * 0.01
        emb_b /= np.linalg.norm(emb_b)
        docs.append(_make_doc(
            doc_id="doc_b_0", embedding=emb_b.tolist(),
            sentiment={"label": "negative", "scores": {"positive": 0.1, "negative": 0.8, "neutral": 0.1}},
        ))

        # Use a Wednesday so no merge
        wednesday = date(2026, 2, 4)

        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository") as MockThemeRepo:
            doc_repo = MockDocRepo.return_value
            theme_repo = MockThemeRepo.return_value

            doc_repo.get_with_embeddings_since = AsyncMock(return_value=docs)
            doc_repo.update_themes = AsyncMock(return_value=True)
            doc_repo.get_sentiments_for_theme = AsyncMock(return_value=[
                {"sentiment": {"label": "positive", "scores": {"positive": 0.8, "negative": 0.1, "neutral": 0.1}},
                 "authority_score": 0.7},
            ])
            theme_repo.get_all = AsyncMock(return_value=[theme_a, theme_b])
            theme_repo.update_centroid = AsyncMock()
            theme_repo.add_metrics = AsyncMock()

            result = await run_daily_clustering(
                mock_database, target_date=wednesday, config=config,
            )

        assert result.documents_fetched == 4
        assert result.documents_assigned == 4
        assert result.documents_unassigned == 0
        assert len(result.errors) == 0
        assert result.elapsed_seconds > 0

        # Themes should have been assigned
        assert doc_repo.update_themes.call_count == 4

    @pytest.mark.asyncio
    async def test_empty_day_returns_zeros(self, config, mock_database):
        """No docs on a given day returns all-zero result."""
        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository"):
            doc_repo = MockDocRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(return_value=[])

            result = await run_daily_clustering(
                mock_database, target_date=date(2026, 2, 5), config=config,
            )

        assert result.documents_fetched == 0
        assert result.documents_assigned == 0
        assert result.documents_unassigned == 0
        assert result.new_themes_created == 0
        assert result.themes_merged == 0
        assert result.metrics_computed == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_fetch_error_captured(self, config, mock_database):
        """DB error during fetch is captured in result.errors."""
        with patch("src.clustering.daily_job.DocumentRepository") as MockDocRepo, \
             patch("src.clustering.daily_job.ThemeRepository"):
            doc_repo = MockDocRepo.return_value
            doc_repo.get_with_embeddings_since = AsyncMock(
                side_effect=RuntimeError("DB connection lost")
            )

            result = await run_daily_clustering(
                mock_database, target_date=date(2026, 2, 5), config=config,
            )

        assert len(result.errors) == 1
        assert "fetch_docs" in result.errors[0]


# ── TestDailyClusteringResult dataclass ───────────────────


class TestDailyClusteringResult:
    """Tests for the DailyClusteringResult dataclass."""

    def test_defaults(self):
        result = DailyClusteringResult(date=date(2026, 2, 5))
        assert result.documents_fetched == 0
        assert result.documents_assigned == 0
        assert result.documents_unassigned == 0
        assert result.new_themes_created == 0
        assert result.themes_merged == 0
        assert result.metrics_computed == 0
        assert result.errors == []
        assert result.elapsed_seconds == 0.0

    def test_errors_mutable_default(self):
        """Ensure errors list is independent per instance."""
        r1 = DailyClusteringResult(date=date(2026, 2, 5))
        r2 = DailyClusteringResult(date=date(2026, 2, 6))
        r1.errors.append("oops")
        assert len(r2.errors) == 0
