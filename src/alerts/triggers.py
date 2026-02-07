"""Stateless trigger functions for alert detection.

Each function checks a single condition against theme metrics and returns
an Alert if the condition is met, or None otherwise. No I/O, no state —
all side effects (dedup, rate limiting, persistence) live in AlertService.

Follows the LifecycleClassifier pattern of pure, trivially testable functions.
"""

from src.alerts.config import AlertConfig
from src.alerts.schemas import Alert
from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import LifecycleTransition


def check_sentiment_velocity(
    theme: Theme,
    today_metrics: ThemeMetrics,
    yesterday_metrics: ThemeMetrics,
    config: AlertConfig,
) -> Alert | None:
    """Check for significant sentiment velocity change between days.

    Fires when abs(today.sentiment_score - yesterday.sentiment_score) exceeds
    the configured threshold. Critical if the delta exceeds the critical
    threshold, warning otherwise.

    Args:
        theme: Theme being analyzed.
        today_metrics: Today's daily metrics.
        yesterday_metrics: Yesterday's daily metrics.
        config: Alert configuration with thresholds.

    Returns:
        Alert or None.
    """
    if today_metrics.sentiment_score is None or yesterday_metrics.sentiment_score is None:
        return None

    delta = today_metrics.sentiment_score - yesterday_metrics.sentiment_score
    abs_delta = abs(delta)

    if abs_delta < config.sentiment_velocity_threshold:
        return None

    direction = "bullish" if delta > 0 else "bearish"

    if abs_delta >= config.sentiment_velocity_critical:
        severity = "critical"
    else:
        severity = "warning"

    return Alert(
        theme_id=theme.theme_id,
        trigger_type="sentiment_velocity",
        severity=severity,
        title=f"Sentiment shift: {theme.name}",
        message=(
            f"Theme '{theme.name}' sentiment shifted {direction} by "
            f"{abs_delta:.2f} ({yesterday_metrics.sentiment_score:.2f} → "
            f"{today_metrics.sentiment_score:.2f})"
        ),
        trigger_data={
            "delta": round(delta, 4),
            "today_score": round(today_metrics.sentiment_score, 4),
            "yesterday_score": round(yesterday_metrics.sentiment_score, 4),
            "direction": direction,
        },
    )


def check_extreme_sentiment(
    theme: Theme,
    metrics: ThemeMetrics,
    config: AlertConfig,
) -> Alert | None:
    """Check for extreme bullish or bearish sentiment crowding.

    Fires when the bullish_ratio exceeds the extreme_bullish_threshold
    or drops below the extreme_bearish_threshold. Always warning severity.

    Args:
        theme: Theme being analyzed.
        metrics: Today's daily metrics.
        config: Alert configuration with thresholds.

    Returns:
        Alert or None.
    """
    if metrics.bullish_ratio is None:
        return None

    if metrics.bullish_ratio > config.extreme_bullish_threshold:
        condition = "extreme_bullish"
        description = f"extremely bullish ({metrics.bullish_ratio:.0%})"
    elif metrics.bullish_ratio < config.extreme_bearish_threshold:
        condition = "extreme_bearish"
        description = f"extremely bearish ({metrics.bullish_ratio:.0%} bullish)"
    else:
        return None

    return Alert(
        theme_id=theme.theme_id,
        trigger_type="extreme_sentiment",
        severity="warning",
        title=f"Extreme sentiment: {theme.name}",
        message=f"Theme '{theme.name}' sentiment is {description}",
        trigger_data={
            "bullish_ratio": round(metrics.bullish_ratio, 4),
            "condition": condition,
        },
    )


def check_volume_surge(
    theme: Theme,
    metrics: ThemeMetrics,
    config: AlertConfig,
) -> Alert | None:
    """Check for abnormal volume z-score.

    Fires when volume_zscore exceeds the surge threshold. Critical if
    it exceeds the critical threshold, warning otherwise.

    Args:
        theme: Theme being analyzed.
        metrics: Today's daily metrics.
        config: Alert configuration with thresholds.

    Returns:
        Alert or None.
    """
    if metrics.volume_zscore is None:
        return None

    if metrics.volume_zscore < config.volume_surge_threshold:
        return None

    if metrics.volume_zscore >= config.volume_surge_critical:
        severity = "critical"
    else:
        severity = "warning"

    return Alert(
        theme_id=theme.theme_id,
        trigger_type="volume_surge",
        severity=severity,
        title=f"Volume surge: {theme.name}",
        message=(
            f"Theme '{theme.name}' volume z-score is "
            f"{metrics.volume_zscore:.1f}σ above normal"
        ),
        trigger_data={
            "volume_zscore": round(metrics.volume_zscore, 4),
            "document_count": metrics.document_count,
        },
    )


def check_lifecycle_change(
    transition: LifecycleTransition,
    theme_name: str,
    config: AlertConfig,
) -> Alert | None:
    """Check if a lifecycle transition warrants an alert.

    Uses the transition's ``is_alertable`` property to decide. Critical
    for emerging→accelerating (theme gaining momentum), warning for others.

    Args:
        transition: Detected lifecycle transition.
        theme_name: Human-readable theme name for the alert message.
        config: Alert configuration (reserved for future threshold tuning).

    Returns:
        Alert or None.
    """
    if not transition.is_alertable:
        return None

    is_gaining = (
        transition.from_stage == "emerging"
        and transition.to_stage == "accelerating"
    )
    severity = "critical" if is_gaining else "warning"

    return Alert(
        theme_id=transition.theme_id,
        trigger_type="lifecycle_change",
        severity=severity,
        title=f"Lifecycle: {theme_name}",
        message=(
            f"Theme '{theme_name}' transitioned from "
            f"{transition.from_stage} → {transition.to_stage}: "
            f"{transition.alert_message}"
        ),
        trigger_data={
            "from_stage": transition.from_stage,
            "to_stage": transition.to_stage,
            "confidence": round(transition.confidence, 4),
            "alert_message": transition.alert_message,
        },
    )


def check_new_theme(
    theme_id: str,
    theme_name: str,
) -> Alert:
    """Create an info alert for a newly detected theme.

    Always fires, always info severity. No config thresholds needed.

    Args:
        theme_id: ID of the new theme.
        theme_name: Human-readable name.

    Returns:
        Alert (always — new themes are always noteworthy).
    """
    return Alert(
        theme_id=theme_id,
        trigger_type="new_theme",
        severity="info",
        title=f"New theme: {theme_name}",
        message=f"New theme '{theme_name}' detected from unassigned documents",
        trigger_data={
            "theme_id": theme_id,
        },
    )


def check_propagated_impact(
    source_theme_name: str,
    target_node_id: str,
    impact: float,
    depth: int,
    config: AlertConfig,
) -> Alert | None:
    """Generate alert for significant propagated sentiment impact.

    Fires when a sentiment change in one theme propagates through the causal
    graph and produces a significant impact on a downstream node. Severity
    is critical if abs(impact) >= 2× threshold, warning otherwise.

    Args:
        source_theme_name: Human-readable name of the source theme.
        target_node_id: ID of the downstream node receiving impact.
        impact: Propagated impact magnitude (can be negative).
        depth: Number of hops from source.
        config: Alert configuration with threshold.

    Returns:
        Alert or None.
    """
    if abs(impact) < config.propagated_impact_threshold:
        return None

    direction = "negative" if impact < 0 else "positive"

    if abs(impact) >= config.propagated_impact_threshold * 2:
        severity = "critical"
    else:
        severity = "warning"

    return Alert(
        theme_id=target_node_id,
        trigger_type="propagated_impact",
        severity=severity,
        title=f"Propagated impact: {target_node_id}",
        message=(
            f"Sentiment change in '{source_theme_name}' propagated "
            f"{direction} impact of {impact:+.3f} to {target_node_id} "
            f"({depth} hop{'s' if depth != 1 else ''} away)"
        ),
        trigger_data={
            "source_theme": source_theme_name,
            "target_node": target_node_id,
            "impact": round(impact, 6),
            "depth": depth,
            "direction": direction,
        },
    )


def check_all_triggers(
    theme: Theme,
    today_metrics: ThemeMetrics,
    yesterday_metrics: ThemeMetrics | None,
    config: AlertConfig,
) -> list[Alert]:
    """Run all metric-based triggers for a single theme.

    Runs sentiment_velocity (if yesterday available), extreme_sentiment,
    and volume_surge. Does NOT run lifecycle_change or new_theme — those
    have different input signatures and are called separately.

    Args:
        theme: Theme being analyzed.
        today_metrics: Today's daily metrics.
        yesterday_metrics: Yesterday's metrics (None skips velocity check).
        config: Alert configuration.

    Returns:
        List of triggered alerts (may be empty).
    """
    alerts: list[Alert] = []

    if yesterday_metrics is not None:
        alert = check_sentiment_velocity(theme, today_metrics, yesterday_metrics, config)
        if alert is not None:
            alerts.append(alert)

    alert = check_extreme_sentiment(theme, today_metrics, config)
    if alert is not None:
        alerts.append(alert)

    alert = check_volume_surge(theme, today_metrics, config)
    if alert is not None:
        alerts.append(alert)

    return alerts
