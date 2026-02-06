"""Volume metrics service for theme-level volume analysis.

Computes normalized volume metrics per theme: platform-weighted volume,
Z-score normalization, velocity (EMA-based), acceleration, and anomaly
detection.  All computation methods are stateless and side-effect-free,
following the LifecycleClassifier pattern for trivial testability.

Duck-typed document input: pure methods accept any object with
``.timestamp``, ``.platform``, and ``.authority_score`` attributes.
"""

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.themes.schemas import ThemeMetrics

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

DEFAULT_PLATFORM_WEIGHTS: dict[str, float] = {
    "twitter": 1.0,
    "reddit": 5.0,
    "news": 20.0,
    "substack": 100.0,
}

VolumeAnomaly = Literal["surge", "collapse"] | None


# ── Config ───────────────────────────────────────────────────


class VolumeMetricsConfig(BaseSettings):
    """Configuration for volume metrics computation.

    All settings can be overridden via ``VOLUME_*`` environment variables
    (e.g., ``VOLUME_DECAY_FACTOR=0.5``).
    """

    model_config = SettingsConfigDict(
        env_prefix="VOLUME_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    decay_factor: float = 0.3
    """Exponential recency decay rate."""

    window_days: int = 7
    """Weighted volume lookback window (days)."""

    history_window: int = 30
    """Z-score rolling history size (days)."""

    min_history_days: int = 7
    """Minimum days of history required for z-score."""

    surge_threshold: float = 3.0
    """Z-score above which a surge anomaly is detected."""

    collapse_threshold: float = -2.0
    """Z-score below which a collapse anomaly is detected."""

    ema_short_span: int = 3
    """Short EMA span for velocity."""

    ema_long_span: int = 7
    """Long EMA span for velocity."""


# ── Service ──────────────────────────────────────────────────


class VolumeMetricsService:
    """Computes normalized volume metrics for a theme.

    Pure computation methods (no DB, stateless):
      - ``compute_weighted_volume`` — platform weight * recency decay * authority
      - ``compute_volume_zscore`` — (current - mean) / std
      - ``compute_velocity`` — short EMA - long EMA
      - ``compute_acceleration`` — delta of last two velocities
      - ``detect_volume_anomaly`` — surge / collapse / None

    Async orchestrator:
      - ``compute_for_theme`` — fetches data, calls all pure methods,
        returns a populated ``ThemeMetrics``.
    """

    def __init__(
        self,
        config: VolumeMetricsConfig | None = None,
        doc_repo: Any = None,
        theme_repo: Any = None,
    ) -> None:
        self._config = config or VolumeMetricsConfig()
        self._doc_repo = doc_repo
        self._theme_repo = theme_repo

    # ── Pure computation methods ─────────────────────────────

    def compute_weighted_volume(
        self,
        docs: list[Any],
        reference_time: datetime | None = None,
    ) -> float:
        """Compute platform-weighted volume with recency decay.

        For each document:
          weight = platform_weight * recency_decay * authority

        where recency_decay = exp(-decay_factor * age_in_days).

        Args:
            docs: Objects with ``.timestamp``, ``.platform``,
                  ``.authority_score`` attributes.
            reference_time: Point in time for recency calculation.
                Defaults to ``datetime.now(UTC)``.

        Returns:
            Sum of weighted contributions. Zero if no documents.
        """
        if not docs:
            return 0.0

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        decay = self._config.decay_factor
        window = timedelta(days=self._config.window_days)
        cutoff = reference_time - window

        total = 0.0
        for doc in docs:
            ts = doc.timestamp
            if ts is None:
                continue
            # Make naive timestamps UTC-aware for comparison
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue

            age_days = (reference_time - ts).total_seconds() / 86400.0
            recency = math.exp(-decay * age_days)

            platform = getattr(doc, "platform", "unknown")
            platform_weight = DEFAULT_PLATFORM_WEIGHTS.get(platform, 1.0)

            authority = getattr(doc, "authority_score", None)
            if authority is None or authority <= 0:
                authority = 0.1  # Floor so docs without authority still count

            total += platform_weight * recency * authority

        return total

    def compute_volume_zscore(
        self,
        current_volume: float,
        rolling_history: list[float],
    ) -> float | None:
        """Compute z-score of current volume against rolling history.

        Args:
            current_volume: Today's weighted volume.
            rolling_history: Previous days' weighted volumes.

        Returns:
            Z-score float, or None if insufficient history.
        """
        if len(rolling_history) < self._config.min_history_days:
            return None

        mean = sum(rolling_history) / len(rolling_history)
        variance = sum((v - mean) ** 2 for v in rolling_history) / len(
            rolling_history
        )
        std = math.sqrt(variance)

        if std == 0:
            return 0.0

        return (current_volume - mean) / std

    def compute_velocity(self, zscores: list[float]) -> float | None:
        """Compute velocity as short EMA minus long EMA of z-scores.

        Positive velocity = accelerating volume. Negative = decelerating.

        Args:
            zscores: Ordered z-score values (oldest first).

        Returns:
            Velocity float, or None if insufficient data.
        """
        short_span = self._config.ema_short_span
        long_span = self._config.ema_long_span

        if len(zscores) < long_span:
            return None

        short_ema = self._ema(zscores, short_span)
        long_ema = self._ema(zscores, long_span)

        return short_ema - long_ema

    def compute_acceleration(self, velocities: list[float]) -> float | None:
        """Compute acceleration as the delta of the last two velocities.

        Args:
            velocities: Ordered velocity values (oldest first).

        Returns:
            Acceleration float, or None if fewer than 2 values.
        """
        if len(velocities) < 2:
            return None
        return velocities[-1] - velocities[-2]

    def detect_volume_anomaly(self, zscore: float | None) -> VolumeAnomaly:
        """Detect volume anomaly from z-score.

        Args:
            zscore: Current z-score (may be None).

        Returns:
            ``"surge"`` if above surge threshold,
            ``"collapse"`` if below collapse threshold,
            ``None`` otherwise.
        """
        if zscore is None:
            return None
        if zscore >= self._config.surge_threshold:
            return "surge"
        if zscore <= self._config.collapse_threshold:
            return "collapse"
        return None

    # ── Helper ───────────────────────────────────────────────

    @staticmethod
    def _ema(values: list[float], span: int) -> float:
        """Compute exponential moving average.

        Uses the standard EMA formula: alpha = 2 / (span + 1).

        Args:
            values: Ordered values (oldest first).
            span: EMA span (period).

        Returns:
            Final EMA value.
        """
        if not values:
            return 0.0

        alpha = 2.0 / (span + 1)
        ema = values[0]
        for v in values[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    # ── Async orchestrator ───────────────────────────────────

    async def compute_for_theme(
        self,
        theme_id: str,
        target_date: date,
    ) -> ThemeMetrics:
        """Orchestrate volume metrics computation for a single theme.

        Fetches documents and historical metrics from repositories,
        runs all pure computation methods, and returns a populated
        ``ThemeMetrics`` instance.

        Args:
            theme_id: Theme identifier.
            target_date: Calendar date to compute metrics for.

        Returns:
            ThemeMetrics with volume fields populated.
        """
        if self._doc_repo is None or self._theme_repo is None:
            raise RuntimeError(
                "doc_repo and theme_repo are required for compute_for_theme"
            )

        # Fetch documents assigned to this theme
        docs = await self._doc_repo.get_documents_by_theme(
            theme_id, limit=500
        )

        # Reference time = end of target_date in UTC
        reference_time = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            23, 59, 59,
            tzinfo=timezone.utc,
        )

        # Compute weighted volume for the target date
        weighted_volume = self.compute_weighted_volume(docs, reference_time)

        # Fetch historical metrics for z-score rolling window
        history_start = target_date - timedelta(
            days=self._config.history_window
        )
        history_end = target_date - timedelta(days=1)
        history = await self._theme_repo.get_metrics_range(
            theme_id, history_start, history_end
        )

        # Build rolling volume history from persisted weighted_volume
        rolling_volumes = [
            m.weighted_volume
            for m in history
            if m.weighted_volume is not None
        ]

        # Z-score
        volume_zscore = self.compute_volume_zscore(
            weighted_volume, rolling_volumes
        )

        # Build z-score history for velocity
        zscores = [
            m.volume_zscore for m in history if m.volume_zscore is not None
        ]
        if volume_zscore is not None:
            zscores.append(volume_zscore)

        velocity = self.compute_velocity(zscores)

        # Build velocity history for acceleration
        velocities = [
            m.velocity for m in history if m.velocity is not None
        ]
        if velocity is not None:
            velocities.append(velocity)

        acceleration = self.compute_acceleration(velocities)

        return ThemeMetrics(
            theme_id=theme_id,
            date=target_date,
            document_count=len(docs),
            weighted_volume=weighted_volume,
            volume_zscore=volume_zscore,
            velocity=velocity,
            acceleration=acceleration,
        )
