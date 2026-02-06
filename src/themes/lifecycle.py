"""Theme lifecycle stage classifier.

Classifies themes into lifecycle stages based on metrics history trends:
- EMERGING: New theme, positive acceleration, low document count
- ACCELERATING: High volume growth, increasing velocity
- MATURE: Peak volume, mainstream coverage, declining novelty
- FADING: Negative acceleration, dropping velocity

Classification uses a 3-day sliding window of ThemeMetrics to compute
velocity and volume trends, then applies rule-based thresholds.
"""

import logging
from typing import Optional

from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import LifecycleTransition

logger = logging.getLogger(__name__)

# ── Tunable thresholds ──────────────────────────────────────

# Minimum metrics data points needed for non-emerging classification.
MIN_HISTORY_DAYS = 3

# Document count ceiling for the "emerging" stage.
EMERGING_DOC_CEILING = 50

# Velocity trend threshold for "accelerating" (positive slope).
ACCELERATING_VELOCITY_THRESHOLD = 0.5

# Velocity trend threshold for "fading" (negative slope).
FADING_VELOCITY_THRESHOLD = -0.3


class LifecycleClassifier:
    """Stateless classifier for theme lifecycle stages.

    Uses recent ThemeMetrics history to classify each theme into one of
    the four lifecycle stages. All classification logic is deterministic
    and side-effect free.
    """

    def classify(
        self,
        theme: Theme,
        metrics_history: list[ThemeMetrics],
    ) -> tuple[str, float]:
        """Classify a theme into a lifecycle stage.

        Args:
            theme: The theme to classify.
            metrics_history: Recent metrics ordered by date ascending.
                Should ideally cover at least 3 days for trend analysis.

        Returns:
            Tuple of (stage, confidence) where stage is one of
            "emerging", "accelerating", "mature", "fading" and
            confidence is 0.0-1.0.
        """
        if len(metrics_history) < MIN_HISTORY_DAYS:
            return "emerging", 0.5  # Low confidence — insufficient data

        recent = metrics_history[-MIN_HISTORY_DAYS:]

        velocities = [m.velocity for m in recent if m.velocity is not None]
        doc_counts = [m.document_count for m in recent]

        velocity_trend = self._compute_trend(velocities) if velocities else 0.0
        volume_trend = self._compute_trend(doc_counts) if doc_counts else 0.0

        # Rule cascade (order matters — most specific first)
        if theme.document_count < EMERGING_DOC_CEILING and velocity_trend > 0:
            confidence = min(1.0, 0.6 + 0.4 * (1 - theme.document_count / EMERGING_DOC_CEILING))
            return "emerging", confidence

        if velocity_trend > ACCELERATING_VELOCITY_THRESHOLD and volume_trend > 0:
            confidence = min(1.0, 0.6 + 0.4 * velocity_trend)
            return "accelerating", confidence

        if velocity_trend < FADING_VELOCITY_THRESHOLD:
            confidence = min(1.0, 0.6 + 0.4 * abs(velocity_trend))
            return "fading", confidence

        # Default: mature (stable volume, no strong trend signals)
        confidence = 0.7
        return "mature", confidence

    def detect_transition(
        self,
        theme: Theme,
        new_stage: str,
        confidence: float = 1.0,
    ) -> Optional[LifecycleTransition]:
        """Detect if a theme changed lifecycle stage.

        Args:
            theme: Theme with its current (old) lifecycle_stage.
            new_stage: Newly classified stage.
            confidence: Classification confidence for the new stage.

        Returns:
            LifecycleTransition if a stage change occurred, None otherwise.
        """
        if theme.lifecycle_stage == new_stage:
            return None

        return LifecycleTransition(
            theme_id=theme.theme_id,
            from_stage=theme.lifecycle_stage,
            to_stage=new_stage,
            confidence=confidence,
        )

    @staticmethod
    def _compute_trend(values: list[float | int]) -> float:
        """Compute a simple linear trend from a short time series.

        Uses least-squares slope normalized by the mean to give a
        scale-independent growth rate. For a constant series, returns 0.0.

        Args:
            values: Ordered numeric values (at least 2).

        Returns:
            Normalized slope. Positive = increasing, negative = decreasing.
        """
        if len(values) < 2:
            return 0.0

        n = len(values)
        # x = 0, 1, 2, ...
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize by mean to get relative change rate
        if y_mean != 0:
            return slope / abs(y_mean)

        return slope
