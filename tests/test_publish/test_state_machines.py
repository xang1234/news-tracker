"""Tests for publish state machine transitions.

These tests validate the transition rules without a database,
exercising the state machine logic directly.
"""

import pytest

from src.contracts.intelligence.db_schemas import (
    VALID_PUBLISH_STATES,
    VALID_RUN_STATUSES,
)
from src.publish.service import (
    PUBLISH_TRANSITIONS,
    RUN_TRANSITIONS,
    _validate_publish_transition,
    _validate_run_transition,
)


class TestRunTransitions:
    """Lane run state machine."""

    def test_all_statuses_have_transition_entry(self) -> None:
        for status in VALID_RUN_STATUSES:
            assert status in RUN_TRANSITIONS

    def test_pending_to_running(self) -> None:
        _validate_run_transition("pending", "running")

    def test_pending_to_cancelled(self) -> None:
        _validate_run_transition("pending", "cancelled")

    def test_running_to_completed(self) -> None:
        _validate_run_transition("running", "completed")

    def test_running_to_failed(self) -> None:
        _validate_run_transition("running", "failed")

    def test_running_to_cancelled(self) -> None:
        _validate_run_transition("running", "cancelled")

    def test_completed_is_terminal(self) -> None:
        for target in VALID_RUN_STATUSES:
            if target != "completed":
                with pytest.raises(ValueError, match="Invalid run transition"):
                    _validate_run_transition("completed", target)

    def test_failed_is_terminal(self) -> None:
        for target in VALID_RUN_STATUSES:
            if target != "failed":
                with pytest.raises(ValueError, match="Invalid run transition"):
                    _validate_run_transition("failed", target)

    def test_cancelled_is_terminal(self) -> None:
        for target in VALID_RUN_STATUSES:
            if target != "cancelled":
                with pytest.raises(ValueError, match="Invalid run transition"):
                    _validate_run_transition("cancelled", target)

    def test_pending_cannot_skip_to_completed(self) -> None:
        with pytest.raises(ValueError, match="Invalid run transition"):
            _validate_run_transition("pending", "completed")

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown current run status"):
            _validate_run_transition("bogus", "running")

    def test_unknown_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown target run status"):
            _validate_run_transition("pending", "bogus")

    def test_no_self_transitions(self) -> None:
        """No status should transition to itself."""
        for status in VALID_RUN_STATUSES:
            if status in RUN_TRANSITIONS[status]:
                pytest.fail(f"Self-transition allowed: {status} → {status}")


class TestPublishTransitions:
    """Published object state machine."""

    def test_all_states_have_transition_entry(self) -> None:
        for state in VALID_PUBLISH_STATES:
            assert state in PUBLISH_TRANSITIONS

    def test_draft_to_review(self) -> None:
        _validate_publish_transition("draft", "review")

    def test_draft_to_published(self) -> None:
        _validate_publish_transition("draft", "published")

    def test_draft_to_retracted(self) -> None:
        _validate_publish_transition("draft", "retracted")

    def test_review_to_published(self) -> None:
        _validate_publish_transition("review", "published")

    def test_review_to_retracted(self) -> None:
        _validate_publish_transition("review", "retracted")

    def test_review_to_draft(self) -> None:
        """Review can be sent back to draft (revise)."""
        _validate_publish_transition("review", "draft")

    def test_published_to_retracted(self) -> None:
        _validate_publish_transition("published", "retracted")

    def test_published_cannot_go_to_draft(self) -> None:
        with pytest.raises(ValueError, match="Invalid publish transition"):
            _validate_publish_transition("published", "draft")

    def test_retracted_is_terminal(self) -> None:
        for target in VALID_PUBLISH_STATES:
            if target != "retracted":
                with pytest.raises(ValueError, match="Invalid publish transition"):
                    _validate_publish_transition("retracted", target)

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown current publish state"):
            _validate_publish_transition("bogus", "published")

    def test_unknown_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown target publish state"):
            _validate_publish_transition("draft", "bogus")
