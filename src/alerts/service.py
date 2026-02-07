"""Alert service orchestrating trigger evaluation, dedup, rate limiting, and persistence.

The sole async component with side effects — Redis for deduplication, DB for
rate limiting and persistence. Trigger logic is delegated to stateless
functions in ``triggers.py``.
"""

import logging
from typing import TYPE_CHECKING, Any

from src.alerts.config import AlertConfig
from src.alerts.repository import AlertRepository
from src.alerts.schemas import Alert
from src.alerts.triggers import (
    check_all_triggers,
    check_lifecycle_change,
    check_new_theme,
)
from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import LifecycleTransition

logger = logging.getLogger(__name__)


class AlertService:
    """Orchestrator for alert generation, filtering, and persistence.

    Combines stateless trigger functions with Redis dedup and DB rate
    limiting, following the orchestrator pattern used by VectorStoreManager.
    """

    def __init__(
        self,
        config: AlertConfig,
        alert_repo: AlertRepository,
        redis_client: Any | None = None,
        dispatcher: Any | None = None,
    ) -> None:
        self._config = config
        self._alert_repo = alert_repo
        self._redis = redis_client
        self._dispatcher = dispatcher

    async def _is_duplicate(self, theme_id: str, trigger_type: str) -> bool:
        """Check Redis SET NX for recent duplicate alerts.

        Key format: ``alert:dedup:{theme_id}:{trigger_type}``
        TTL: ``config.dedup_ttl_hours`` hours.

        Returns False (not a duplicate) if Redis is unavailable — graceful
        degradation means we'd rather have duplicates than silence.

        Args:
            theme_id: Theme identifier.
            trigger_type: Alert trigger type.

        Returns:
            True if a recent alert exists for this combination.
        """
        if self._redis is None:
            return False

        key = f"alert:dedup:{theme_id}:{trigger_type}"
        ttl_seconds = self._config.dedup_ttl_hours * 3600

        try:
            was_set = await self._redis.set(key, "1", nx=True, ex=ttl_seconds)
            # SET NX returns True if the key was set (no duplicate),
            # None/False if it already existed (duplicate).
            return not was_set
        except Exception as e:
            logger.warning("Redis dedup check failed, allowing alert: %s", e)
            return False

    async def _is_rate_limited(self, severity: str) -> bool:
        """Check if the daily rate limit for this severity is exhausted.

        Uses ``AlertRepository.count_today_by_severity()`` for the count.
        Limit of 0 means unlimited.

        Args:
            severity: Alert severity level.

        Returns:
            True if rate limited (should suppress).
        """
        limit_map = {
            "critical": self._config.daily_limit_critical,
            "warning": self._config.daily_limit_warning,
            "info": self._config.daily_limit_info,
        }
        daily_limit = limit_map.get(severity, 0)

        if daily_limit == 0:
            return False

        try:
            count = await self._alert_repo.count_today_by_severity(severity)
            return count >= daily_limit
        except Exception as e:
            logger.warning("Rate limit check failed, allowing alert: %s", e)
            return False

    async def _filter_alert(self, alert: Alert) -> bool:
        """Apply dedup and rate limiting to an alert.

        Args:
            alert: Candidate alert.

        Returns:
            True if the alert should be persisted (passed all filters).
        """
        if await self._is_duplicate(alert.theme_id, alert.trigger_type):
            logger.debug(
                "Alert deduplicated: %s/%s", alert.theme_id, alert.trigger_type,
            )
            return False

        if await self._is_rate_limited(alert.severity):
            logger.debug(
                "Alert rate limited: %s severity", alert.severity,
            )
            return False

        return True

    async def generate_alerts(
        self,
        themes: list[Theme],
        today_metrics_map: dict[str, ThemeMetrics],
        yesterday_metrics_map: dict[str, ThemeMetrics],
        lifecycle_transitions: list[LifecycleTransition] | None = None,
        new_theme_ids: list[str] | None = None,
    ) -> list[Alert]:
        """Main entry point: evaluate all triggers, filter, and persist.

        Called by Phase 12 of the daily clustering job.

        Args:
            themes: All current themes.
            today_metrics_map: Today's metrics keyed by theme_id.
            yesterday_metrics_map: Yesterday's metrics keyed by theme_id.
            lifecycle_transitions: Detected transitions from Phase 11.
            new_theme_ids: Theme IDs created in Phase 9.

        Returns:
            List of persisted alerts.
        """
        lifecycle_transitions = lifecycle_transitions or []
        new_theme_ids = new_theme_ids or []

        candidates: list[Alert] = []

        # Metric-based triggers per theme
        theme_map = {t.theme_id: t for t in themes}
        for theme in themes:
            today = today_metrics_map.get(theme.theme_id)
            if today is None:
                continue

            yesterday = yesterday_metrics_map.get(theme.theme_id)
            alerts = check_all_triggers(theme, today, yesterday, self._config)
            candidates.extend(alerts)

        # Lifecycle transition triggers
        for transition in lifecycle_transitions:
            theme = theme_map.get(transition.theme_id)
            theme_name = theme.name if theme else transition.theme_id
            alert = check_lifecycle_change(transition, theme_name, self._config)
            if alert is not None:
                candidates.append(alert)

        # New theme triggers
        for tid in new_theme_ids:
            theme = theme_map.get(tid)
            theme_name = theme.name if theme else tid
            candidates.append(check_new_theme(tid, theme_name))

        # Filter: dedup + rate limit
        filtered: list[Alert] = []
        for alert in candidates:
            if await self._filter_alert(alert):
                filtered.append(alert)

        # Persist
        if filtered:
            persisted = await self._alert_repo.create_batch(filtered)
            logger.info(
                "Alerts generated: %d candidates, %d filtered, %d persisted",
                len(candidates),
                len(filtered),
                len(persisted),
            )

            # Dispatch notifications (never blocks alert persistence)
            if self._dispatcher is not None and persisted:
                try:
                    await self._dispatcher.dispatch_batch(persisted)
                except Exception as e:
                    logger.error("Notification dispatch failed: %s", e)

            return persisted

        logger.info(
            "Alert generation: %d candidates, all filtered out", len(candidates),
        )
        return []
