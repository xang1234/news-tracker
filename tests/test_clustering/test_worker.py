"""Tests for ClusteringWorker real-time theme assignment."""

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pytest

from src.clustering.config import ClusteringConfig
from src.clustering.worker import ClusteringWorker


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def config():
    return ClusteringConfig(
        similarity_threshold_assign=0.75,
        centroid_learning_rate=0.01,
    )


@pytest.fixture
def mock_queue():
    q = AsyncMock()
    q.ack = AsyncMock()
    q.nack = AsyncMock()
    q.connect = AsyncMock()
    q.close = AsyncMock()
    q.health_check = AsyncMock(return_value=True)
    q.get_pending_count = AsyncMock(return_value=0)
    return q


@pytest.fixture
def mock_database():
    db = AsyncMock()
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.execute = AsyncMock()
    db.health_check = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_doc_repo():
    return AsyncMock()


@pytest.fixture
def mock_theme_repo():
    return AsyncMock()


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock()
    r.close = AsyncMock()
    return r


@pytest.fixture
def worker(config, mock_queue, mock_database, mock_doc_repo, mock_theme_repo, mock_redis):
    w = ClusteringWorker(
        queue=mock_queue,
        database=mock_database,
        config=config,
        batch_size=10,
    )
    w._doc_repo = mock_doc_repo
    w._theme_repo = mock_theme_repo
    w._redis = mock_redis
    return w


def _make_job(doc_id="doc_001", model="finbert", msg_id="msg-1"):
    """Create a ClusteringJob-like object."""
    from src.clustering.queue import ClusteringJob
    return ClusteringJob(
        document_id=doc_id,
        embedding_model=model,
        message_id=msg_id,
    )


def _make_doc(doc_id="doc_001", platform="twitter", embedding=None, embedding_minilm=None):
    """Create a mock NormalizedDocument."""
    doc = MagicMock()
    doc.id = doc_id
    doc.platform = platform
    doc.embedding = embedding
    doc.embedding_minilm = embedding_minilm
    return doc


def _make_theme(theme_id="theme_abc", centroid=None, doc_count=10):
    """Create a mock Theme."""
    theme = MagicMock()
    theme.theme_id = theme_id
    theme.centroid = centroid if centroid is not None else np.random.randn(768).astype(np.float32)
    theme.document_count = doc_count
    return theme


# ── TestEMACentroidUpdate ──────────────────────────────────


class TestEMACentroidUpdate:
    """Tests for the static EMA centroid update method."""

    def test_basic_update(self):
        centroid = np.ones(768, dtype=np.float32)
        embedding = np.zeros(768, dtype=np.float32)
        result = ClusteringWorker._ema_centroid_update(centroid, embedding, 0.1)
        expected = 0.9 * centroid + 0.1 * embedding
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        assert result.dtype == np.float32

    def test_zero_learning_rate(self):
        """lr→0: centroid unchanged (clamped by config, but test edge)."""
        centroid = np.ones(768, dtype=np.float32)
        embedding = np.full(768, 99.0, dtype=np.float32)
        # lr=0 means no update — use a tiny value
        result = ClusteringWorker._ema_centroid_update(centroid, embedding, 1e-10)
        np.testing.assert_allclose(result, centroid, atol=1e-6)

    def test_full_learning_rate(self):
        """lr=1: centroid replaced by embedding."""
        centroid = np.ones(768, dtype=np.float32)
        embedding = np.full(768, 5.0, dtype=np.float32)
        result = ClusteringWorker._ema_centroid_update(centroid, embedding, 1.0)
        np.testing.assert_allclose(result, embedding, rtol=1e-5)

    def test_output_dtype_float32(self):
        centroid = np.ones(768, dtype=np.float64)
        embedding = np.ones(768, dtype=np.float64)
        result = ClusteringWorker._ema_centroid_update(centroid, embedding, 0.5)
        assert result.dtype == np.float32


# ── TestGetEmbedding ───────────────────────────────────────


class TestGetEmbedding:
    """Tests for _get_embedding model dispatch."""

    def test_finbert_returns_embedding(self):
        doc = _make_doc(embedding=[1.0, 2.0, 3.0])
        result = ClusteringWorker._get_embedding(doc, "finbert")
        assert result == [1.0, 2.0, 3.0]

    def test_minilm_returns_embedding_minilm(self):
        doc = _make_doc(embedding_minilm=[4.0, 5.0, 6.0])
        result = ClusteringWorker._get_embedding(doc, "minilm")
        assert result == [4.0, 5.0, 6.0]

    def test_missing_finbert_returns_none(self):
        doc = _make_doc(embedding=None)
        result = ClusteringWorker._get_embedding(doc, "finbert")
        assert result is None

    def test_unknown_model_defaults_to_finbert(self):
        doc = _make_doc(embedding=[7.0, 8.0])
        result = ClusteringWorker._get_embedding(doc, "unknown_model")
        assert result == [7.0, 8.0]


# ── TestProcessBatch ──────────────────────────────────────


class TestProcessBatch:
    """Tests for _process_batch with various scenarios."""

    @pytest.mark.asyncio
    async def test_successful_assignment(self, worker, mock_doc_repo, mock_theme_repo, mock_queue, mock_database):
        """Happy path: doc has embedding, matches a theme."""
        embedding = np.random.randn(768).astype(np.float32).tolist()
        doc = _make_doc(embedding=embedding)
        theme = _make_theme()
        similarity = 0.82

        mock_doc_repo.get_by_id.return_value = doc
        mock_theme_repo.find_similar.return_value = [(theme, similarity)]
        mock_theme_repo.update_centroid.return_value = None
        mock_doc_repo.update_themes.return_value = True

        job = _make_job()
        await worker._process_batch([job])

        # Document theme_ids updated
        mock_doc_repo.update_themes.assert_called_once_with(
            "doc_001", [theme.theme_id]
        )
        # Centroid EMA updated
        mock_theme_repo.update_centroid.assert_called_once()
        # Atomic document_count increment
        mock_database.execute.assert_called_once()
        call_args = mock_database.execute.call_args
        assert "document_count + 1" in call_args[0][0]
        # Job acknowledged
        mock_queue.ack.assert_called_once_with("msg-1")

    @pytest.mark.asyncio
    async def test_no_matching_theme(self, worker, mock_doc_repo, mock_theme_repo, mock_queue):
        """No theme above threshold — skip without error."""
        embedding = np.random.randn(768).astype(np.float32).tolist()
        doc = _make_doc(embedding=embedding)
        mock_doc_repo.get_by_id.return_value = doc
        mock_theme_repo.find_similar.return_value = []

        job = _make_job()
        await worker._process_batch([job])

        mock_doc_repo.update_themes.assert_not_called()
        mock_queue.ack.assert_called_once_with("msg-1")

    @pytest.mark.asyncio
    async def test_idempotency_skip(self, worker, mock_redis, mock_doc_repo, mock_queue):
        """Redis SET NX returns False — already processed."""
        mock_redis.set.return_value = False  # Already set

        job = _make_job()
        await worker._process_batch([job])

        mock_doc_repo.get_by_id.assert_not_called()
        mock_queue.ack.assert_called_once_with("msg-1")

    @pytest.mark.asyncio
    async def test_missing_document(self, worker, mock_doc_repo, mock_queue):
        """Document not in DB — skip."""
        mock_doc_repo.get_by_id.return_value = None

        job = _make_job()
        await worker._process_batch([job])

        mock_queue.ack.assert_called_once_with("msg-1")

    @pytest.mark.asyncio
    async def test_missing_embedding_clears_idem_key(self, worker, mock_doc_repo, mock_redis, mock_queue):
        """Doc exists but no embedding — clear idem key for retry."""
        doc = _make_doc(embedding=None)
        mock_doc_repo.get_by_id.return_value = doc

        job = _make_job()
        await worker._process_batch([job])

        # Idempotency key should be deleted so re-queue succeeds
        mock_redis.delete.assert_called_once()
        mock_queue.ack.assert_called_once_with("msg-1")

    @pytest.mark.asyncio
    async def test_minilm_skip(self, worker, mock_doc_repo, mock_queue):
        """MiniLM-embedded docs are skipped (centroids are 768-dim)."""
        doc = _make_doc(embedding_minilm=[1.0] * 384)
        mock_doc_repo.get_by_id.return_value = doc

        job = _make_job(model="minilm")
        await worker._process_batch([job])

        mock_queue.ack.assert_called_once_with("msg-1")

    @pytest.mark.asyncio
    async def test_error_nacks_job(self, worker, mock_doc_repo, mock_theme_repo, mock_queue):
        """Exception during processing — nack the job."""
        embedding = np.random.randn(768).astype(np.float32).tolist()
        doc = _make_doc(embedding=embedding)
        mock_doc_repo.get_by_id.return_value = doc
        mock_theme_repo.find_similar.side_effect = RuntimeError("DB down")

        job = _make_job()
        await worker._process_batch([job])

        mock_queue.nack.assert_called_once()
        mock_queue.ack.assert_not_called()


# ── TestHealthCheck ────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, worker, mock_queue, mock_database):
        worker._running = True
        result = await worker.health_check()
        assert result["running"] is True
        assert result["queue_healthy"] is True
        assert result["database_healthy"] is True


# ── TestStopSignal ─────────────────────────────────────────


class TestStopSignal:
    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, worker):
        worker._running = True
        await worker.stop()
        assert worker.is_running is False

    def test_is_running_property(self, worker):
        assert worker.is_running is False
        worker._running = True
        assert worker.is_running is True
