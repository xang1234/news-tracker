"""Schema definitions for narrative momentum."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

VALID_RUN_STATUSES = frozenset({"active", "cooling", "closed"})
VALID_SIGNAL_STATES = frozenset({"inactive", "active"})


@dataclass
class NarrativeRun:
    """Persisted narrative run."""

    run_id: str
    theme_id: str
    status: str
    centroid: np.ndarray
    label: str
    started_at: datetime
    last_document_at: datetime
    closed_at: datetime | None = None
    doc_count: int = 0
    platform_first_seen: dict[str, str] = field(default_factory=dict)
    ticker_counts: dict[str, int] = field(default_factory=dict)
    avg_sentiment: float = 0.0
    avg_authority: float = 0.0
    platform_count: int = 0
    current_rate_per_hour: float = 0.0
    current_acceleration: float = 0.0
    conviction_score: float = 0.0
    last_signal_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.status not in VALID_RUN_STATUSES:
            raise ValueError(
                f"Invalid narrative run status {self.status!r}. "
                f"Must be one of {sorted(VALID_RUN_STATUSES)}"
            )


@dataclass
class NarrativeRunBucket:
    """Five-minute narrative run bucket."""

    run_id: str
    bucket_start: datetime
    doc_count: int = 0
    platform_counts: dict[str, int] = field(default_factory=dict)
    ticker_counts: dict[str, int] = field(default_factory=dict)
    sentiment_sum: float = 0.0
    sentiment_weight: float = 0.0
    sentiment_confidence_sum: float = 0.0
    sentiment_doc_count: int = 0
    authority_sum: float = 0.0
    high_authority_sentiment_sum: float = 0.0
    high_authority_weight: float = 0.0
    high_authority_doc_count: int = 0
    low_authority_sentiment_sum: float = 0.0
    low_authority_weight: float = 0.0
    low_authority_doc_count: int = 0


@dataclass
class NarrativeSignalState:
    """Persisted signal hysteresis state."""

    run_id: str
    trigger_type: str
    state: str = "inactive"
    last_score: float = 0.0
    last_alert_at: datetime | None = None
    last_transition_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cooldown_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in VALID_SIGNAL_STATES:
            raise ValueError(
                f"Invalid signal state {self.state!r}. "
                f"Must be one of {sorted(VALID_SIGNAL_STATES)}"
            )
