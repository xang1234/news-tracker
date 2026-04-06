"""Replay, quarantine, and operator inspection hooks.

Provides the minimum operator-oriented functions needed to
inspect failures, quarantine lanes, and plan replays without
turning news-tracker into a second product UI.

Three hook types:
    - Replay: build a plan to re-run a failed lane with the same config
    - Quarantine: block or unblock a lane from publication
    - Inspection: trace a failure back to runs, artifacts, and quality

All functions are stateless decision helpers. The caller
handles persistence (via PublishService, repositories, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.contracts.intelligence.db_schemas import LaneRun
from src.publish.lane_health import QuarantineRecord, QuarantineState

# Terminal run statuses — runs that can be replayed or inspected.
# Derived from RUN_TRANSITIONS in publish/service.py (states with
# no outgoing transitions).
TERMINAL_RUN_STATUSES = frozenset({"failed", "completed", "cancelled"})


# -- Replay plan ---------------------------------------------------------------


@dataclass(frozen=True)
class ReplayPlan:
    """Plan to re-run a failed or problematic lane run.

    Does NOT execute — the caller uses the plan to create a new
    run via PublishService.create_run(). This separation keeps
    replays auditable.

    Attributes:
        source_run_id: The original run being replayed.
        lane: Which lane to replay.
        config_snapshot: The original run's frozen config.
        reason: Why the replay is needed.
        requested_by: Who/what requested the replay.
        requested_at: When the replay was requested.
        metadata: Extra context (error details, related manifests, etc.).
    """

    source_run_id: str
    lane: str
    config_snapshot: dict[str, Any]
    reason: str
    requested_by: str = "operator"
    requested_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_run_id": self.source_run_id,
            "lane": self.lane,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "config_keys": sorted(self.config_snapshot.keys()),
        }


# -- Quarantine actions --------------------------------------------------------

VALID_QUARANTINE_ACTIONS = frozenset({"quarantine", "watch", "lift"})


@dataclass(frozen=True)
class QuarantineAction:
    """An operator action to quarantine, watch, or lift a lane.

    Attributes:
        lane: Target lane.
        action: "quarantine" (block), "watch" (observe), or "lift" (unblock).
        reason: Why the action is being taken.
        actor: Who is taking the action (operator name or system).
        acted_at: When the action was taken.
        metadata: Extra context.
    """

    lane: str
    action: str
    reason: str
    actor: str = "operator"
    acted_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in VALID_QUARANTINE_ACTIONS:
            raise ValueError(
                f"Invalid quarantine action {self.action!r}. "
                f"Must be one of {sorted(VALID_QUARANTINE_ACTIONS)}"
            )

    def to_quarantine_record(self) -> QuarantineRecord | None:
        """Convert to a QuarantineRecord for persistence.

        Returns None for "lift" actions (caller should clear the record).
        """
        if self.action == "lift":
            return None
        state = (
            QuarantineState.QUARANTINED
            if self.action == "quarantine"
            else QuarantineState.WATCH
        )
        return QuarantineRecord(
            lane=self.lane,
            reason=self.reason,
            quarantined_at=self.acted_at,
            quarantined_by=self.actor,
            state=state,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "action": self.action,
            "reason": self.reason,
            "actor": self.actor,
            "acted_at": self.acted_at.isoformat(),
        }


# -- Inspection report ---------------------------------------------------------


@dataclass(frozen=True)
class RunFailureContext:
    """Context about a failed run for operator inspection.

    Attributes:
        run_id: The failed run.
        lane: Which lane.
        status: Run status (typically "failed").
        error_message: Error from the run.
        started_at: When the run started.
        completed_at: When it failed.
        config_snapshot: Frozen config for reproducibility.
        metrics: Run-level metrics at failure time.
    """

    run_id: str
    lane: str
    status: str
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectionReport:
    """Operator inspection report tracing a failure to artifacts.

    Links a failed or problematic run to related artifacts so
    operators can understand what went wrong without manually
    joining tables.

    Attributes:
        run_context: The failed run's context.
        dead_letter_count: Claims that were dead-lettered during this run.
        review_task_count: Review tasks created during this run.
        manifest_ids: Manifests associated with this run.
        quality_issues: Quality metric failures related to this run.
        replay_plan: Pre-built replay plan (if the run is replayable).
        inspected_at: When the inspection was performed.
    """

    run_context: RunFailureContext
    dead_letter_count: int = 0
    review_task_count: int = 0
    manifest_ids: list[str] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)
    replay_plan: ReplayPlan | None = None
    inspected_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_context.run_id,
            "lane": self.run_context.lane,
            "status": self.run_context.status,
            "error_message": self.run_context.error_message,
            "dead_letter_count": self.dead_letter_count,
            "review_task_count": self.review_task_count,
            "manifest_ids": self.manifest_ids,
            "quality_issues": self.quality_issues,
            "has_replay_plan": self.replay_plan is not None,
            "inspected_at": self.inspected_at.isoformat(),
        }


# -- Hook functions (stateless) ------------------------------------------------


def build_replay_plan(
    run: LaneRun,
    reason: str,
    *,
    requested_by: str = "operator",
    now: datetime | None = None,
) -> ReplayPlan:
    """Build a replay plan from a failed lane run.

    The plan captures the original config_snapshot so the replay
    uses the same configuration. The caller executes the plan
    by creating a new run via PublishService.

    Args:
        run: The failed or problematic lane run.
        reason: Why a replay is needed.
        requested_by: Who is requesting (operator name or system).
        now: Timestamp for the request.

    Returns:
        ReplayPlan ready for execution.

    Raises:
        ValueError: If the run is not in a terminal state.
    """
    if now is None:
        now = datetime.now(UTC)

    if run.status not in TERMINAL_RUN_STATUSES:
        raise ValueError(
            f"Cannot replay run {run.run_id}: status is {run.status!r}, "
            f"not a terminal state"
        )

    return ReplayPlan(
        source_run_id=run.run_id,
        lane=run.lane,
        config_snapshot=dict(run.config_snapshot),
        reason=reason,
        requested_by=requested_by,
        requested_at=now,
        metadata={
            "original_status": run.status,
            "original_error": run.error_message,
        },
    )


def build_inspection_report(
    run: LaneRun,
    *,
    dead_letter_count: int = 0,
    review_task_count: int = 0,
    manifest_ids: list[str] | None = None,
    quality_issues: list[str] | None = None,
    include_replay_plan: bool = True,
    now: datetime | None = None,
) -> InspectionReport:
    """Build an inspection report for a failed or problematic run.

    The caller provides pre-aggregated counts from the relevant
    repositories. The hook builds the report with a replay plan
    if the run is in a terminal state.

    Args:
        run: The run to inspect.
        dead_letter_count: Claims dead-lettered during this run.
        review_task_count: Review tasks from this run.
        manifest_ids: Manifests associated with this run.
        quality_issues: Quality metric failure descriptions.
        include_replay_plan: Whether to build a replay plan.
        now: Inspection timestamp.

    Returns:
        InspectionReport with linked context and optional replay plan.
    """
    if now is None:
        now = datetime.now(UTC)

    context = RunFailureContext(
        run_id=run.run_id,
        lane=run.lane,
        status=run.status,
        error_message=run.error_message,
        started_at=run.started_at,
        completed_at=run.completed_at,
        config_snapshot=dict(run.config_snapshot),
        metrics=dict(run.metrics),
    )

    replay = None
    if include_replay_plan and run.status in TERMINAL_RUN_STATUSES:
        replay = build_replay_plan(
            run, reason="Operator inspection replay", now=now,
        )

    return InspectionReport(
        run_context=context,
        dead_letter_count=dead_letter_count,
        review_task_count=review_task_count,
        manifest_ids=manifest_ids or [],
        quality_issues=quality_issues or [],
        replay_plan=replay,
        inspected_at=now,
    )
