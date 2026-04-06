"""Tests for replay, quarantine, and operator inspection hooks.

Verifies replay plan building, quarantine actions, inspection
reports, and the linkage between failures and artifacts.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.contracts.intelligence.db_schemas import LaneRun
from src.monitoring.operator_hooks import (
    QuarantineAction,
    build_inspection_report,
    build_replay_plan,
)
from src.publish.lane_health import QuarantineState

NOW = datetime(2026, 4, 1, tzinfo=UTC)


# -- Helpers ---------------------------------------------------------------


def _failed_run(
    run_id: str = "run_001",
    lane: str = "narrative",
    error: str = "Timeout connecting to provider",
    config: dict | None = None,
) -> LaneRun:
    return LaneRun(
        run_id=run_id,
        lane=lane,
        status="failed",
        contract_version="0.1.0",
        started_at=NOW - timedelta(hours=1),
        completed_at=NOW,
        error_message=error,
        config_snapshot=config or {"batch_size": 100},
        metrics={"doc_count": 42},
    )


def _completed_run(lane: str = "filing") -> LaneRun:
    return LaneRun(
        run_id="run_002",
        lane=lane,
        status="completed",
        contract_version="0.1.0",
        started_at=NOW - timedelta(hours=2),
        completed_at=NOW - timedelta(hours=1),
        config_snapshot={"provider": "edgartools"},
    )


def _running_run() -> LaneRun:
    return LaneRun(
        run_id="run_003",
        lane="narrative",
        status="running",
        contract_version="0.1.0",
        started_at=NOW - timedelta(minutes=30),
    )


# -- Replay plan tests -----------------------------------------------------


class TestReplayPlan:
    """Build replay plans from failed runs."""

    def test_basic_replay(self) -> None:
        plan = build_replay_plan(_failed_run(), "Provider timeout", now=NOW)
        assert plan.source_run_id == "run_001"
        assert plan.lane == "narrative"
        assert plan.reason == "Provider timeout"
        assert plan.config_snapshot == {"batch_size": 100}
        assert plan.requested_by == "operator"

    def test_preserves_config(self) -> None:
        config = {"batch_size": 200, "timeout": 30}
        run = _failed_run(config=config)
        plan = build_replay_plan(run, "retry", now=NOW)
        assert plan.config_snapshot == config
        assert plan.config_snapshot is not run.config_snapshot

    def test_completed_run_replayable(self) -> None:
        plan = build_replay_plan(_completed_run(), "Re-evaluate", now=NOW)
        assert plan.source_run_id == "run_002"

    def test_cancelled_run_replayable(self) -> None:
        run = LaneRun(run_id="run_c", lane="narrative", status="cancelled",
                      contract_version="0.1.0")
        plan = build_replay_plan(run, "Cancelled too early", now=NOW)
        assert plan.source_run_id == "run_c"

    def test_running_run_not_replayable(self) -> None:
        with pytest.raises(ValueError, match="not a terminal state"):
            build_replay_plan(_running_run(), "impatient", now=NOW)

    def test_pending_run_not_replayable(self) -> None:
        run = LaneRun(run_id="run_p", lane="narrative", status="pending",
                      contract_version="0.1.0")
        with pytest.raises(ValueError, match="not a terminal state"):
            build_replay_plan(run, "too early", now=NOW)

    def test_metadata_includes_original_error(self) -> None:
        plan = build_replay_plan(_failed_run(), "retry", now=NOW)
        assert plan.metadata["original_error"] == "Timeout connecting to provider"
        assert plan.metadata["original_status"] == "failed"

    def test_custom_requester(self) -> None:
        plan = build_replay_plan(
            _failed_run(), "automated retry",
            requested_by="retry_worker", now=NOW,
        )
        assert plan.requested_by == "retry_worker"

    def test_to_dict(self) -> None:
        plan = build_replay_plan(_failed_run(), "test", now=NOW)
        d = plan.to_dict()
        assert d["source_run_id"] == "run_001"
        assert d["lane"] == "narrative"
        assert "config_keys" in d

    def test_frozen(self) -> None:
        plan = build_replay_plan(_failed_run(), "test", now=NOW)
        with pytest.raises(AttributeError):
            plan.reason = "changed"  # type: ignore[misc]


# -- Quarantine action tests -----------------------------------------------


class TestQuarantineAction:
    """Operator quarantine/watch/lift actions."""

    def test_quarantine(self) -> None:
        action = QuarantineAction(
            lane="narrative", action="quarantine",
            reason="Dead-letter rate too high",
        )
        assert action.action == "quarantine"
        record = action.to_quarantine_record()
        assert record is not None
        assert record.state == QuarantineState.QUARANTINED
        assert record.reason == "Dead-letter rate too high"

    def test_watch(self) -> None:
        action = QuarantineAction(
            lane="filing", action="watch",
            reason="Provider intermittent",
        )
        record = action.to_quarantine_record()
        assert record is not None
        assert record.state == QuarantineState.WATCH

    def test_lift(self) -> None:
        action = QuarantineAction(
            lane="narrative", action="lift",
            reason="Issue resolved",
        )
        record = action.to_quarantine_record()
        assert record is None

    def test_invalid_action(self) -> None:
        with pytest.raises(ValueError, match="Invalid quarantine action"):
            QuarantineAction(
                lane="narrative", action="pause",
                reason="not a valid action",
            )

    def test_valid_actions(self) -> None:
        for a in ("quarantine", "watch", "lift"):
            action = QuarantineAction(lane="narrative", action=a, reason="test")
            assert action.action == a

    def test_custom_actor(self) -> None:
        action = QuarantineAction(
            lane="filing", action="quarantine",
            reason="test", actor="admin_user",
        )
        record = action.to_quarantine_record()
        assert record is not None
        assert record.quarantined_by == "admin_user"

    def test_metadata_passed_through(self) -> None:
        action = QuarantineAction(
            lane="narrative", action="quarantine",
            reason="test", metadata={"ticket": "INC-123"},
        )
        record = action.to_quarantine_record()
        assert record is not None
        assert record.metadata == {"ticket": "INC-123"}

    def test_to_dict(self) -> None:
        action = QuarantineAction(
            lane="narrative", action="quarantine",
            reason="high error rate", actor="ops",
            acted_at=NOW,
        )
        d = action.to_dict()
        assert d["lane"] == "narrative"
        assert d["action"] == "quarantine"
        assert d["actor"] == "ops"

    def test_frozen(self) -> None:
        action = QuarantineAction(
            lane="narrative", action="quarantine", reason="test",
        )
        with pytest.raises(AttributeError):
            action.reason = "changed"  # type: ignore[misc]


# -- Inspection report tests -----------------------------------------------


class TestInspectionReport:
    """Operator inspection of failed runs."""

    def test_basic_inspection(self) -> None:
        report = build_inspection_report(
            _failed_run(),
            dead_letter_count=5,
            review_task_count=2,
            manifest_ids=["manifest_001"],
            quality_issues=["Lineage below 85%"],
            now=NOW,
        )
        assert report.run_context.run_id == "run_001"
        assert report.run_context.lane == "narrative"
        assert report.run_context.status == "failed"
        assert report.dead_letter_count == 5
        assert report.review_task_count == 2
        assert report.manifest_ids == ["manifest_001"]
        assert report.quality_issues == ["Lineage below 85%"]

    def test_includes_replay_plan(self) -> None:
        report = build_inspection_report(_failed_run(), now=NOW)
        assert report.replay_plan is not None
        assert report.replay_plan.source_run_id == "run_001"

    def test_skip_replay_plan(self) -> None:
        report = build_inspection_report(
            _failed_run(), include_replay_plan=False, now=NOW,
        )
        assert report.replay_plan is None

    def test_running_run_no_replay(self) -> None:
        """Running runs can be inspected but not replayed."""
        report = build_inspection_report(
            _running_run(), include_replay_plan=True, now=NOW,
        )
        assert report.replay_plan is None

    def test_completed_run_inspectable(self) -> None:
        report = build_inspection_report(_completed_run(), now=NOW)
        assert report.run_context.status == "completed"
        assert report.replay_plan is not None

    def test_context_preserves_config(self) -> None:
        report = build_inspection_report(
            _failed_run(config={"key": "value"}), now=NOW,
        )
        assert report.run_context.config_snapshot == {"key": "value"}

    def test_context_preserves_metrics(self) -> None:
        report = build_inspection_report(_failed_run(), now=NOW)
        assert report.run_context.metrics == {"doc_count": 42}

    def test_empty_artifacts(self) -> None:
        report = build_inspection_report(_failed_run(), now=NOW)
        assert report.dead_letter_count == 0
        assert report.review_task_count == 0
        assert report.manifest_ids == []
        assert report.quality_issues == []

    def test_to_dict(self) -> None:
        report = build_inspection_report(
            _failed_run(),
            dead_letter_count=3,
            now=NOW,
        )
        d = report.to_dict()
        assert d["run_id"] == "run_001"
        assert d["dead_letter_count"] == 3
        assert d["has_replay_plan"] is True
        assert isinstance(d["inspected_at"], str)

    def test_frozen(self) -> None:
        report = build_inspection_report(_failed_run(), now=NOW)
        with pytest.raises(AttributeError):
            report.dead_letter_count = 0  # type: ignore[misc]


# -- Realistic scenario tests -----------------------------------------------


class TestRealisticScenarios:
    """End-to-end operator workflows."""

    def test_inspect_quarantine_replay_flow(self) -> None:
        """Operator inspects failure, quarantines, then replays."""
        # 1. Inspect the failure
        report = build_inspection_report(
            _failed_run(error="Provider rate limited"),
            dead_letter_count=12,
            quality_issues=["Stale evidence above 25%"],
            now=NOW,
        )
        assert report.run_context.error_message == "Provider rate limited"
        assert report.dead_letter_count == 12

        # 2. Quarantine the lane
        action = QuarantineAction(
            lane="narrative",
            action="quarantine",
            reason="Provider rate limited, dead letters accumulating",
            actor="oncall_sre",
            acted_at=NOW,
        )
        record = action.to_quarantine_record()
        assert record is not None
        assert record.state == QuarantineState.QUARANTINED

        # 3. After fixing, build replay plan
        plan = report.replay_plan
        assert plan is not None
        assert plan.config_snapshot == {"batch_size": 100}

        # 4. Lift quarantine
        lift = QuarantineAction(
            lane="narrative", action="lift",
            reason="Provider recovered, replay successful",
            actor="oncall_sre",
        )
        assert lift.to_quarantine_record() is None
