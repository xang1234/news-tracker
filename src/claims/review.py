"""Review task schemas for the claim review queue.

Review tasks are backend state that captures ambiguous or high-impact
claim outcomes requiring human or automated review. Every task links
back to concrete claims and concepts so decisions can flow into
recompute paths.

Task types:
    - entity_review: Ambiguous entity resolution needs confirmation
    - claim_review: Low-confidence or contradictory claim needs review
    - merge_proposal: Two concepts may be the same entity
    - split_proposal: One concept may actually be two entities
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# -- Valid state sets -------------------------------------------------------

VALID_TASK_TYPES = frozenset(
    {"entity_review", "claim_review", "merge_proposal", "split_proposal"}
)

VALID_TASK_STATUSES = frozenset(
    {"pending", "assigned", "resolved", "dismissed"}
)

VALID_TRIGGER_REASONS = frozenset(
    {
        "low_confidence",
        "close_alternatives",
        "llm_proposed",
        "contradiction",
        "high_impact_predicate",
        "manual",
    }
)

VALID_RESOLUTIONS = frozenset(
    {"approved", "rejected", "merged", "split", "deferred"}
)

# State machine: from → set of allowed targets
REVIEW_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"assigned", "resolved", "dismissed"}),
    "assigned": frozenset({"resolved", "dismissed", "pending"}),
    "resolved": frozenset(),  # terminal
    "dismissed": frozenset(),  # terminal
}


def make_review_task_id(
    task_type: str,
    claim_ids: list[str],
    concept_ids: list[str],
) -> str:
    """Generate a deterministic review task ID.

    Same trigger inputs produce the same task ID, enabling idempotent
    creation. A claim_review for the same claims won't create duplicates.
    """
    parts = [
        task_type,
        ",".join(sorted(claim_ids)),
        ",".join(sorted(concept_ids)),
    ]
    key_input = "\x00".join(parts)
    return f"review_{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"


# -- ReviewTask -------------------------------------------------------------


@dataclass
class ReviewTask:
    """A review work item in the claim review queue.

    Attributes:
        task_id: Deterministic ID (from make_review_task_id).
        task_type: Kind of review (entity_review, merge_proposal, etc.).
        status: Lifecycle state (pending → assigned → resolved/dismissed).
        trigger_reason: What triggered this review task.
        claim_ids: Related evidence claim IDs.
        concept_ids: Related concept IDs.
        priority: Numeric priority (0=critical, 4=backlog).
        assigned_to: Who is reviewing (None if unassigned).
        resolution: What was decided (None until resolved).
        resolution_notes: Free-text notes from the reviewer.
        payload: Task-type-specific structured data.
        lineage: Recompute metadata (affected runs, downstream claims).
        metadata: Extensible metadata.
    """

    task_id: str
    task_type: str
    trigger_reason: str
    status: str = "pending"
    claim_ids: list[str] = field(default_factory=list)
    concept_ids: list[str] = field(default_factory=list)
    priority: int = 2
    assigned_to: str | None = None
    resolution: str | None = None
    resolution_notes: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type {self.task_type!r}. "
                f"Must be one of {sorted(VALID_TASK_TYPES)}"
            )
        if self.status not in VALID_TASK_STATUSES:
            raise ValueError(
                f"Invalid task status {self.status!r}. "
                f"Must be one of {sorted(VALID_TASK_STATUSES)}"
            )
        if self.trigger_reason not in VALID_TRIGGER_REASONS:
            raise ValueError(
                f"Invalid trigger_reason {self.trigger_reason!r}. "
                f"Must be one of {sorted(VALID_TRIGGER_REASONS)}"
            )
        if self.resolution is not None and self.resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution {self.resolution!r}. "
                f"Must be one of {sorted(VALID_RESOLUTIONS)}"
            )


def validate_review_transition(current: str, target: str) -> None:
    """Validate a review task state transition.

    Raises ValueError if the transition is not allowed.
    """
    if current not in REVIEW_TRANSITIONS:
        raise ValueError(f"Unknown review status {current!r}")
    allowed = REVIEW_TRANSITIONS[current]
    if target not in allowed:
        raise ValueError(
            f"Invalid review transition: {current!r} → {target!r}. "
            f"Allowed from {current!r}: {sorted(allowed) or 'none (terminal)'}"
        )
