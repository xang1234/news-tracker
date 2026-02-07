"""Drift detection service.

Runs four independent checks against the existing database tables
(documents, themes, theme_metrics) to detect model degradation,
distribution shift, clustering fragmentation, and sentiment drift.

Follows the BacktestEngine pattern: takes Database directly, uses
module-level pure helpers for testability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.monitoring.config import DriftConfig
from src.monitoring.schemas import DriftReport, DriftResult, DriftSeverity
from src.storage.database import Database

logger = logging.getLogger(__name__)


# ── Pure helpers (stateless, no I/O) ─────────────────────────


def _classify_severity(
    value: float,
    warning: float,
    critical: float,
) -> DriftSeverity:
    """Classify a metric value into a severity level.

    Args:
        value: The measured value (higher = worse).
        warning: Threshold for warning.
        critical: Threshold for critical.

    Returns:
        DriftSeverity: ok, warning, or critical.
    """
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(P||Q) with Laplace smoothing.

    Adds a small constant to both distributions to avoid log(0),
    then re-normalises to valid probability distributions.

    Args:
        p: Baseline distribution (must sum > 0).
        q: Comparison distribution (must sum > 0).

    Returns:
        Non-negative float.  KL divergence is 0 iff p == q.
    """
    eps = 1e-10
    p_smooth = p.astype(np.float64) + eps
    q_smooth = q.astype(np.float64) + eps

    p_norm = p_smooth / p_smooth.sum()
    q_norm = q_smooth / q_smooth.sum()

    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def _parse_embedding(value: Any) -> np.ndarray | None:
    """Parse a pgvector string, list, or ndarray into float32 array.

    Handles the same formats as ThemeRepository._parse_centroid
    but returns None for NULL/None values instead of raising.

    Args:
        value: pgvector string, list of floats, ndarray, or None.

    Returns:
        Float32 numpy array, or None if value is None/empty.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, list):
        return np.array(value, dtype=np.float32)
    if isinstance(value, str):
        stripped = value.strip("[]")
        if not stripped:
            return None
        return np.array(
            [float(x) for x in stripped.split(",")],
            dtype=np.float32,
        )
    return None


# ── Service ──────────────────────────────────────────────────


class DriftService:
    """Detects distribution drift and model degradation.

    Args:
        database: Async database connection.
        config: Optional drift configuration (defaults created if None).
    """

    def __init__(
        self,
        database: Database,
        config: DriftConfig | None = None,
    ) -> None:
        self._db = database
        self._config = config or DriftConfig()

    # ── Individual checks ────────────────────────────────────

    async def check_embedding_drift(self) -> DriftResult:
        """Compare L2-norm distribution of recent vs baseline embeddings.

        Samples up to ``embedding_sample_size`` documents per window,
        computes histograms of embedding L2 norms, and measures KL
        divergence between them.
        """
        cfg = self._config
        now = datetime.now(timezone.utc)
        recent_since = now - timedelta(hours=cfg.embedding_recent_hours)
        baseline_since = now - timedelta(days=cfg.embedding_baseline_days)

        # Fetch baseline embeddings
        baseline_rows = await self._db.fetch(
            "SELECT embedding FROM documents "
            "WHERE embedding IS NOT NULL AND fetched_at >= $1 AND fetched_at < $2 "
            "ORDER BY RANDOM() LIMIT $3",
            baseline_since,
            recent_since,
            cfg.embedding_sample_size,
        )

        # Fetch recent embeddings
        recent_rows = await self._db.fetch(
            "SELECT embedding FROM documents "
            "WHERE embedding IS NOT NULL AND fetched_at >= $1 "
            "ORDER BY RANDOM() LIMIT $2",
            recent_since,
            cfg.embedding_sample_size,
        )

        if len(baseline_rows) < 10 or len(recent_rows) < 10:
            return DriftResult(
                drift_type="embedding_drift",
                severity="ok",
                value=0.0,
                thresholds={
                    "warning": cfg.embedding_kl_warning,
                    "critical": cfg.embedding_kl_critical,
                },
                message=(
                    f"Insufficient data for embedding drift check "
                    f"(baseline={len(baseline_rows)}, recent={len(recent_rows)})"
                ),
                metadata={
                    "baseline_count": len(baseline_rows),
                    "recent_count": len(recent_rows),
                },
            )

        # Compute L2 norms
        baseline_norms = []
        for row in baseline_rows:
            emb = _parse_embedding(row["embedding"])
            if emb is not None:
                baseline_norms.append(float(np.linalg.norm(emb)))

        recent_norms = []
        for row in recent_rows:
            emb = _parse_embedding(row["embedding"])
            if emb is not None:
                recent_norms.append(float(np.linalg.norm(emb)))

        if len(baseline_norms) < 10 or len(recent_norms) < 10:
            return DriftResult(
                drift_type="embedding_drift",
                severity="ok",
                value=0.0,
                thresholds={
                    "warning": cfg.embedding_kl_warning,
                    "critical": cfg.embedding_kl_critical,
                },
                message="Insufficient valid embeddings for drift check",
                metadata={
                    "baseline_valid": len(baseline_norms),
                    "recent_valid": len(recent_norms),
                },
            )

        # Shared bin edges for comparable histograms
        all_norms = baseline_norms + recent_norms
        bin_edges = np.linspace(
            min(all_norms), max(all_norms), cfg.embedding_num_bins + 1
        )

        baseline_hist, _ = np.histogram(
            baseline_norms, bins=bin_edges, density=True
        )
        recent_hist, _ = np.histogram(
            recent_norms, bins=bin_edges, density=True
        )

        kl = _kl_divergence(baseline_hist, recent_hist)
        severity = _classify_severity(
            kl, cfg.embedding_kl_warning, cfg.embedding_kl_critical
        )

        return DriftResult(
            drift_type="embedding_drift",
            severity=severity,
            value=round(kl, 6),
            thresholds={
                "warning": cfg.embedding_kl_warning,
                "critical": cfg.embedding_kl_critical,
            },
            message=(
                f"Embedding L2 norm KL divergence: {kl:.4f} "
                f"(baseline={len(baseline_norms)}, recent={len(recent_norms)})"
            ),
            metadata={
                "baseline_count": len(baseline_norms),
                "recent_count": len(recent_norms),
                "baseline_mean_norm": round(float(np.mean(baseline_norms)), 4),
                "recent_mean_norm": round(float(np.mean(recent_norms)), 4),
            },
        )

    async def check_theme_fragmentation(self) -> DriftResult:
        """Check if theme creation rate is abnormally high.

        Counts themes created per day over the lookback window and
        flags if the most recent day exceeds thresholds.
        """
        cfg = self._config
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=cfg.fragmentation_lookback_days)

        rows = await self._db.fetch(
            "SELECT DATE(created_at) AS day, COUNT(*) AS cnt "
            "FROM themes "
            "WHERE created_at >= $1 AND deleted_at IS NULL "
            "GROUP BY DATE(created_at) "
            "ORDER BY day",
            since,
        )

        if not rows:
            return DriftResult(
                drift_type="theme_fragmentation",
                severity="ok",
                value=0.0,
                thresholds={
                    "warning": float(cfg.fragmentation_warning),
                    "critical": float(cfg.fragmentation_critical),
                },
                message="No themes created in lookback window",
                metadata={"lookback_days": cfg.fragmentation_lookback_days},
            )

        # Use the most recent day's count for severity
        latest_count = int(rows[-1]["cnt"])
        severity = _classify_severity(
            float(latest_count),
            float(cfg.fragmentation_warning),
            float(cfg.fragmentation_critical),
        )

        daily_counts = {str(row["day"]): int(row["cnt"]) for row in rows}

        return DriftResult(
            drift_type="theme_fragmentation",
            severity=severity,
            value=float(latest_count),
            thresholds={
                "warning": float(cfg.fragmentation_warning),
                "critical": float(cfg.fragmentation_critical),
            },
            message=(
                f"Theme creation rate: {latest_count} on latest day "
                f"(normal range: {cfg.fragmentation_normal_min}-{cfg.fragmentation_normal_max})"
            ),
            metadata={
                "latest_day": str(rows[-1]["day"]),
                "daily_counts": daily_counts,
            },
        )

    async def check_sentiment_calibration(self) -> DriftResult:
        """Check if sentiment distribution has shifted.

        Computes daily average bullish_ratio from theme_metrics,
        then calculates the z-score of the most recent day against
        the baseline window.
        """
        cfg = self._config
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=cfg.sentiment_baseline_days)

        rows = await self._db.fetch(
            "SELECT date, AVG(bullish_ratio) AS avg_ratio "
            "FROM theme_metrics "
            "WHERE date >= $1 AND bullish_ratio IS NOT NULL "
            "GROUP BY date "
            "ORDER BY date",
            since.date(),
        )

        if len(rows) < 3:
            return DriftResult(
                drift_type="sentiment_calibration",
                severity="ok",
                value=0.0,
                thresholds={
                    "warning": cfg.sentiment_zscore_warning,
                    "critical": cfg.sentiment_zscore_critical,
                },
                message=(
                    f"Insufficient data for sentiment calibration "
                    f"({len(rows)} day(s) available, need at least 3)"
                ),
                metadata={"days_available": len(rows)},
            )

        ratios = [float(row["avg_ratio"]) for row in rows]
        baseline = ratios[:-1]
        latest = ratios[-1]

        mean = float(np.mean(baseline))
        std = float(np.std(baseline))

        if std < 1e-10:
            zscore = 0.0
        else:
            zscore = abs(latest - mean) / std

        severity = _classify_severity(
            zscore, cfg.sentiment_zscore_warning, cfg.sentiment_zscore_critical
        )

        return DriftResult(
            drift_type="sentiment_calibration",
            severity=severity,
            value=round(zscore, 4),
            thresholds={
                "warning": cfg.sentiment_zscore_warning,
                "critical": cfg.sentiment_zscore_critical,
            },
            message=(
                f"Sentiment bullish ratio z-score: {zscore:.2f} "
                f"(latest={latest:.3f}, baseline mean={mean:.3f})"
            ),
            metadata={
                "latest_ratio": round(latest, 4),
                "baseline_mean": round(mean, 4),
                "baseline_std": round(std, 4),
                "latest_date": str(rows[-1]["date"]),
            },
        )

    async def check_cluster_stability(self) -> DriftResult:
        """Check if theme centroids match their assigned documents.

        For each active theme with a stored centroid, computes the mean
        embedding of recently assigned documents and measures cosine
        distance to the stored centroid.  Reports the average distance
        across all themes.
        """
        cfg = self._config
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=cfg.stability_lookback_days)

        # Get active themes with centroids
        themes = await self._db.fetch(
            "SELECT theme_id, centroid FROM themes "
            "WHERE centroid IS NOT NULL AND deleted_at IS NULL",
        )

        if not themes:
            return DriftResult(
                drift_type="cluster_stability",
                severity="ok",
                value=0.0,
                thresholds={
                    "warning": cfg.stability_warning,
                    "critical": cfg.stability_critical,
                },
                message="No themes with centroids found",
                metadata={"theme_count": 0},
            )

        distances: list[float] = []
        themes_checked = 0

        for theme_row in themes:
            centroid = _parse_embedding(theme_row["centroid"])
            if centroid is None:
                continue

            # Get recent docs assigned to this theme
            doc_rows = await self._db.fetch(
                "SELECT embedding FROM documents "
                "WHERE $1 = ANY(theme_ids) "
                "AND embedding IS NOT NULL "
                "AND fetched_at >= $2 "
                "LIMIT 100",
                theme_row["theme_id"],
                since,
            )

            if len(doc_rows) < 3:
                continue

            doc_embeddings = []
            for dr in doc_rows:
                emb = _parse_embedding(dr["embedding"])
                if emb is not None:
                    doc_embeddings.append(emb)

            if len(doc_embeddings) < 3:
                continue

            # Mean doc embedding
            mean_emb = np.mean(doc_embeddings, axis=0)

            # Cosine distance = 1 - cosine_similarity
            c_norm = np.linalg.norm(centroid)
            m_norm = np.linalg.norm(mean_emb)
            if c_norm < 1e-10 or m_norm < 1e-10:
                continue

            cosine_sim = float(np.dot(centroid, mean_emb) / (c_norm * m_norm))
            cosine_dist = 1.0 - cosine_sim
            distances.append(cosine_dist)
            themes_checked += 1

        if not distances:
            return DriftResult(
                drift_type="cluster_stability",
                severity="ok",
                value=0.0,
                thresholds={
                    "warning": cfg.stability_warning,
                    "critical": cfg.stability_critical,
                },
                message="No themes with enough recent documents to check stability",
                metadata={"themes_available": len(themes), "themes_checked": 0},
            )

        avg_distance = float(np.mean(distances))
        severity = _classify_severity(
            avg_distance, cfg.stability_warning, cfg.stability_critical
        )

        return DriftResult(
            drift_type="cluster_stability",
            severity=severity,
            value=round(avg_distance, 6),
            thresholds={
                "warning": cfg.stability_warning,
                "critical": cfg.stability_critical,
            },
            message=(
                f"Average centroid drift: {avg_distance:.4f} cosine distance "
                f"({themes_checked} themes checked)"
            ),
            metadata={
                "themes_checked": themes_checked,
                "max_distance": round(float(max(distances)), 6),
                "min_distance": round(float(min(distances)), 6),
            },
        )

    # ── Composite checks ─────────────────────────────────────

    async def run_quick_check(self) -> DriftReport:
        """Run embedding drift check only (for hourly cron)."""
        report = DriftReport()
        try:
            result = await self.check_embedding_drift()
            report.results.append(result)
        except Exception:
            logger.exception("Embedding drift check failed")
            report.results.append(
                DriftResult(
                    drift_type="embedding_drift",
                    severity="warning",
                    value=-1.0,
                    thresholds={},
                    message="Embedding drift check failed with exception",
                )
            )
        return report

    async def run_daily_check(self) -> DriftReport:
        """Run all four checks (phase-resilient).

        Each check runs independently — a failure in one does not
        prevent the others from executing.
        """
        report = DriftReport()

        checks = [
            self.check_embedding_drift,
            self.check_theme_fragmentation,
            self.check_sentiment_calibration,
            self.check_cluster_stability,
        ]

        for check in checks:
            try:
                result = await check()
                report.results.append(result)
            except Exception:
                # Derive drift_type from method name: check_X -> X
                name = check.__name__.replace("check_", "")
                logger.exception(f"Drift check '{name}' failed")
                report.results.append(
                    DriftResult(
                        drift_type=name if name in {
                            "embedding_drift",
                            "theme_fragmentation",
                            "sentiment_calibration",
                            "cluster_stability",
                        } else "embedding_drift",
                        severity="warning",
                        value=-1.0,
                        thresholds={},
                        message=f"Drift check '{name}' failed with exception",
                    )
                )

        return report

    async def run_weekly_report(self) -> DriftReport:
        """Run a weekly report (delegates to daily, extensible later)."""
        return await self.run_daily_check()
