"""Tests for drift detection and monitoring service.

Mocks Database with AsyncMock to avoid real DB connections.
Tests are grouped by check type following the project convention.
"""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.monitoring.config import DriftConfig
from src.monitoring.schemas import (
    VALID_DRIFT_TYPES,
    DriftReport,
    DriftResult,
    DriftSeverity,
)
from src.monitoring.service import (
    DriftService,
    _classify_severity,
    _kl_divergence,
    _parse_embedding,
)


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Create a mock Database with async fetch methods."""
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchval = AsyncMock(return_value=None)
    return db


@pytest.fixture
def config():
    """Create a default DriftConfig."""
    return DriftConfig()


@pytest.fixture
def service(mock_db, config):
    """Create a DriftService with mock database."""
    return DriftService(mock_db, config)


# ── Helper: _classify_severity ───────────────────────────────


class TestClassifySeverity:
    def test_ok_below_warning(self):
        assert _classify_severity(0.05, 0.1, 0.2) == "ok"

    def test_warning_at_threshold(self):
        assert _classify_severity(0.1, 0.1, 0.2) == "warning"

    def test_warning_between(self):
        assert _classify_severity(0.15, 0.1, 0.2) == "warning"

    def test_critical_at_threshold(self):
        assert _classify_severity(0.2, 0.1, 0.2) == "critical"

    def test_critical_above(self):
        assert _classify_severity(0.5, 0.1, 0.2) == "critical"

    def test_zero_is_ok(self):
        assert _classify_severity(0.0, 0.1, 0.2) == "ok"


# ── Helper: _kl_divergence ───────────────────────────────────


class TestKLDivergence:
    def test_identical_distributions(self):
        p = np.array([0.2, 0.3, 0.5])
        kl = _kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=1e-8)

    def test_divergent_distributions(self):
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.1, 0.8])
        kl = _kl_divergence(p, q)
        assert kl > 0.1

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            p = rng.random(20)
            q = rng.random(20)
            assert _kl_divergence(p, q) >= 0.0

    def test_zero_bins_handled(self):
        """Laplace smoothing prevents log(0)."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        kl = _kl_divergence(p, q)
        assert np.isfinite(kl)
        assert kl > 0


# ── Helper: _parse_embedding ─────────────────────────────────


class TestParseEmbedding:
    def test_none_returns_none(self):
        assert _parse_embedding(None) is None

    def test_ndarray_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = _parse_embedding(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, arr)

    def test_list_conversion(self):
        result = _parse_embedding([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_pgvector_string(self):
        result = _parse_embedding("[0.1,0.2,0.3]")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == pytest.approx(0.1)

    def test_empty_string_returns_none(self):
        assert _parse_embedding("[]") is None

    def test_unknown_type_returns_none(self):
        assert _parse_embedding(42) is None


# ── Check: Embedding Drift ───────────────────────────────────


class TestEmbeddingDrift:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, service, mock_db):
        """Returns ok with message when not enough documents."""
        mock_db.fetch = AsyncMock(return_value=[])
        result = await service.check_embedding_drift()
        assert result.severity == "ok"
        assert "Insufficient" in result.message

    @pytest.mark.asyncio
    async def test_similar_distributions(self, service, mock_db, config):
        """Similar embeddings produce low KL divergence."""
        # Reduce bins to prevent sparse histogram artifacts with 1000 samples
        config.embedding_num_bins = 20

        rng = np.random.default_rng(42)
        # Both windows drawn from identical distribution
        baseline_embs = rng.normal(0.0, 1.0, size=(500, 32)).astype(np.float32)
        recent_embs = rng.normal(0.0, 1.0, size=(500, 32)).astype(np.float32)

        baseline_rows = [{"embedding": e.tolist()} for e in baseline_embs]
        recent_rows = [{"embedding": e.tolist()} for e in recent_embs]

        mock_db.fetch = AsyncMock(side_effect=[baseline_rows, recent_rows])

        result = await service.check_embedding_drift()
        assert result.drift_type == "embedding_drift"
        assert result.severity == "ok"
        assert result.value < 0.1

    @pytest.mark.asyncio
    async def test_divergent_distributions(self, service, mock_db):
        """Very different embeddings produce high KL divergence."""
        rng = np.random.default_rng(42)
        # Baseline: norm ~ 10, Recent: norm ~ 50 (huge shift)
        baseline_embs = rng.normal(10.0, 0.5, size=(50, 768)).astype(np.float32)
        recent_embs = rng.normal(50.0, 0.5, size=(20, 768)).astype(np.float32)

        baseline_rows = [{"embedding": e.tolist()} for e in baseline_embs]
        recent_rows = [{"embedding": e.tolist()} for e in recent_embs]

        mock_db.fetch = AsyncMock(side_effect=[baseline_rows, recent_rows])

        result = await service.check_embedding_drift()
        assert result.drift_type == "embedding_drift"
        assert result.severity == "critical"
        assert result.value > 0.2


# ── Check: Theme Fragmentation ───────────────────────────────


class TestThemeFragmentation:
    @pytest.mark.asyncio
    async def test_no_themes(self, service, mock_db):
        """No themes in window returns ok."""
        mock_db.fetch = AsyncMock(return_value=[])
        result = await service.check_theme_fragmentation()
        assert result.severity == "ok"
        assert result.value == 0.0

    @pytest.mark.asyncio
    async def test_normal_rate(self, service, mock_db):
        """Normal theme creation rate is ok."""
        mock_db.fetch = AsyncMock(return_value=[
            {"day": date(2026, 2, 5), "cnt": 8},
            {"day": date(2026, 2, 6), "cnt": 10},
        ])
        result = await service.check_theme_fragmentation()
        assert result.drift_type == "theme_fragmentation"
        assert result.severity == "ok"
        assert result.value == 10.0

    @pytest.mark.asyncio
    async def test_high_rate_critical(self, service, mock_db):
        """50+ themes per day triggers critical."""
        mock_db.fetch = AsyncMock(return_value=[
            {"day": date(2026, 2, 5), "cnt": 10},
            {"day": date(2026, 2, 6), "cnt": 55},
        ])
        result = await service.check_theme_fragmentation()
        assert result.severity == "critical"
        assert result.value == 55.0

    @pytest.mark.asyncio
    async def test_warning_rate(self, service, mock_db):
        """30-49 themes per day triggers warning."""
        mock_db.fetch = AsyncMock(return_value=[
            {"day": date(2026, 2, 6), "cnt": 35},
        ])
        result = await service.check_theme_fragmentation()
        assert result.severity == "warning"


# ── Check: Sentiment Calibration ─────────────────────────────


class TestSentimentCalibration:
    @pytest.mark.asyncio
    async def test_insufficient_data(self, service, mock_db):
        """Returns ok when < 3 days available."""
        mock_db.fetch = AsyncMock(return_value=[
            {"date": date(2026, 2, 6), "avg_ratio": 0.5},
        ])
        result = await service.check_sentiment_calibration()
        assert result.severity == "ok"
        assert "Insufficient" in result.message

    @pytest.mark.asyncio
    async def test_stable_sentiment(self, service, mock_db):
        """Stable bullish_ratio has low z-score."""
        # Wider baseline spread so 0.50 on last day is well within 2σ
        mock_db.fetch = AsyncMock(return_value=[
            {"date": date(2026, 2, 1), "avg_ratio": 0.45},
            {"date": date(2026, 2, 2), "avg_ratio": 0.55},
            {"date": date(2026, 2, 3), "avg_ratio": 0.48},
            {"date": date(2026, 2, 4), "avg_ratio": 0.52},
            {"date": date(2026, 2, 5), "avg_ratio": 0.50},
        ])
        result = await service.check_sentiment_calibration()
        assert result.drift_type == "sentiment_calibration"
        assert result.severity == "ok"

    @pytest.mark.asyncio
    async def test_extreme_shift(self, service, mock_db):
        """Sudden jump in bullish ratio triggers warning or critical."""
        mock_db.fetch = AsyncMock(return_value=[
            {"date": date(2026, 2, 1), "avg_ratio": 0.50},
            {"date": date(2026, 2, 2), "avg_ratio": 0.51},
            {"date": date(2026, 2, 3), "avg_ratio": 0.49},
            {"date": date(2026, 2, 4), "avg_ratio": 0.50},
            # Extreme jump on last day
            {"date": date(2026, 2, 5), "avg_ratio": 0.95},
        ])
        result = await service.check_sentiment_calibration()
        assert result.severity in ("warning", "critical")
        assert result.value > 2.0


# ── Check: Cluster Stability ────────────────────────────────


class TestClusterStability:
    @pytest.mark.asyncio
    async def test_no_themes(self, service, mock_db):
        """No themes returns ok."""
        mock_db.fetch = AsyncMock(return_value=[])
        result = await service.check_cluster_stability()
        assert result.severity == "ok"
        assert "No themes" in result.message

    @pytest.mark.asyncio
    async def test_stable_clusters(self, service, mock_db):
        """Centroid close to doc mean = low distance."""
        rng = np.random.default_rng(42)
        centroid = rng.normal(0, 1, size=768).astype(np.float32)
        # Docs clustered very close to centroid
        doc_embs = [
            (centroid + rng.normal(0, 0.01, size=768).astype(np.float32)).tolist()
            for _ in range(10)
        ]

        theme_rows = [{"theme_id": "t1", "centroid": centroid.tolist()}]
        doc_rows = [{"embedding": e} for e in doc_embs]

        mock_db.fetch = AsyncMock(side_effect=[theme_rows, doc_rows])

        result = await service.check_cluster_stability()
        assert result.drift_type == "cluster_stability"
        assert result.severity == "ok"
        assert result.value < 0.01


# ── Drift Report composition ────────────────────────────────


class TestDriftReport:
    def test_empty_report_is_ok(self):
        report = DriftReport()
        assert report.overall_severity == "ok"
        assert not report.has_issues

    def test_overall_worst_of(self):
        report = DriftReport(results=[
            DriftResult(
                drift_type="embedding_drift", severity="ok",
                value=0.01, thresholds={}, message="ok",
            ),
            DriftResult(
                drift_type="theme_fragmentation", severity="critical",
                value=60.0, thresholds={}, message="high",
            ),
        ])
        assert report.overall_severity == "critical"
        assert report.has_issues

    @pytest.mark.asyncio
    async def test_quick_returns_one_result(self, service):
        report = await service.run_quick_check()
        assert len(report.results) == 1
        assert report.results[0].drift_type == "embedding_drift"

    @pytest.mark.asyncio
    async def test_daily_returns_four_results(self, service, mock_db):
        """Daily check always produces 4 results (phase-resilient)."""
        mock_db.fetch = AsyncMock(return_value=[])
        report = await service.run_daily_check()
        assert len(report.results) == 4

    @pytest.mark.asyncio
    async def test_phase_resilience(self, mock_db, config):
        """A failing check doesn't prevent other checks from running."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on the 2nd fetch call (inside theme_fragmentation)
            if call_count == 3:
                raise RuntimeError("DB timeout")
            return []

        mock_db.fetch = AsyncMock(side_effect=side_effect)
        svc = DriftService(mock_db, config)
        report = await svc.run_daily_check()

        # All 4 results should be present (failed ones get warning placeholders)
        assert len(report.results) == 4

    def test_drift_result_validation(self):
        """Invalid drift_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid drift_type"):
            DriftResult(
                drift_type="invalid_type",
                severity="ok",
                value=0.0,
                thresholds={},
                message="test",
            )

    def test_valid_drift_types(self):
        """All four drift types are in VALID_DRIFT_TYPES."""
        assert len(VALID_DRIFT_TYPES) == 4
        assert "embedding_drift" in VALID_DRIFT_TYPES
        assert "theme_fragmentation" in VALID_DRIFT_TYPES
        assert "sentiment_calibration" in VALID_DRIFT_TYPES
        assert "cluster_stability" in VALID_DRIFT_TYPES
