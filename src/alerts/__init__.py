"""Alert service for detecting actionable conditions in theme metrics.

Components:
- Alert: Dataclass mapping to the alerts table
- AlertConfig: Pydantic settings for thresholds and rate limits
- AlertRepository: CRUD operations for alert persistence
- AlertService: Orchestrator for trigger evaluation, dedup, and persistence
- AlertTriggerType / AlertSeverity: Literal types for type safety
- VALID_TRIGGER_TYPES / VALID_SEVERITIES: Frozensets for runtime validation
- NotificationChannel / WebhookChannel / SlackChannel: Delivery channels
- CircuitBreaker: Resilience wrapper for channels
- NotificationConfig / NotificationDispatcher: Dispatch orchestration
"""

from src.alerts.channels import (
    CircuitBreaker,
    NotificationChannel,
    SlackChannel,
    WebhookChannel,
)
from src.alerts.config import AlertConfig
from src.alerts.dispatcher import NotificationConfig, NotificationDispatcher
from src.alerts.repository import AlertRepository
from src.alerts.schemas import (
    VALID_SEVERITIES,
    VALID_TRIGGER_TYPES,
    Alert,
    AlertSeverity,
    AlertTriggerType,
)
from src.alerts.service import AlertService

__all__ = [
    "Alert",
    "AlertConfig",
    "AlertRepository",
    "AlertService",
    "AlertSeverity",
    "AlertTriggerType",
    "CircuitBreaker",
    "NotificationChannel",
    "NotificationConfig",
    "NotificationDispatcher",
    "SlackChannel",
    "VALID_SEVERITIES",
    "VALID_TRIGGER_TYPES",
    "WebhookChannel",
]
