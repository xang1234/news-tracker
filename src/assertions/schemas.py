"""Schema definitions for resolved assertions.

Resolved assertions are the aggregation layer between raw evidence
claims and downstream consumers (graph, scoring, publishing). Each
assertion represents a stable current-belief about a subject-predicate-
object triple, backed by explicit support and contradiction links.

Assertion ID is deterministic: sha256(subject_concept_id + predicate +
object_concept_id). Multiple claims about the same triple contribute
to the same assertion via claim links.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# -- Valid state sets -------------------------------------------------------

VALID_ASSERTION_STATUSES = frozenset(
    {"active", "disputed", "retracted", "superseded"}
)

VALID_LINK_TYPES = frozenset({"support", "contradiction"})


def make_assertion_id(
    subject_concept_id: str,
    predicate: str,
    object_concept_id: str | None = None,
) -> str:
    """Generate a deterministic assertion ID from the triple.

    Same triple always produces the same ID — claims about the
    same relationship feed into a single assertion.
    """
    parts = [
        subject_concept_id,
        predicate.lower().strip(),
        object_concept_id or "",
    ]
    key_input = "\x00".join(parts)
    return f"asrt_{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"


# -- ResolvedAssertion ------------------------------------------------------


@dataclass
class ResolvedAssertion:
    """A stable current-belief object aggregated from evidence claims.

    Attributes:
        assertion_id: Deterministic ID from the triple.
        subject_concept_id: The subject concept.
        predicate: What is asserted (e.g., "supplies_to").
        object_concept_id: The object concept (None for unary assertions).
        confidence: Aggregate confidence from supporting evidence.
        status: Lifecycle state (active, disputed, retracted, superseded).
        valid_from: When the asserted fact became true.
        valid_to: When the asserted fact ceased to be true (None = ongoing).
        support_count: Number of supporting claim links.
        contradiction_count: Number of contradicting claim links.
        first_seen_at: Earliest supporting claim timestamp.
        last_evidence_at: Most recent evidence claim timestamp.
        source_diversity: Count of distinct source types contributing.
        metadata: Extensible metadata.
    """

    assertion_id: str
    subject_concept_id: str
    predicate: str
    object_concept_id: str | None = None
    confidence: float = 0.0
    status: str = "active"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    support_count: int = 0
    contradiction_count: int = 0
    first_seen_at: datetime | None = None
    last_evidence_at: datetime | None = None
    source_diversity: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.status not in VALID_ASSERTION_STATUSES:
            raise ValueError(
                f"Invalid assertion status {self.status!r}. "
                f"Must be one of {sorted(VALID_ASSERTION_STATUSES)}"
            )
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_from > self.valid_to
        ):
            raise ValueError(
                f"valid_from ({self.valid_from}) must be <= "
                f"valid_to ({self.valid_to})"
            )

    @property
    def is_disputed(self) -> bool:
        """Whether the assertion has any contradicting evidence."""
        return self.contradiction_count > 0

    @property
    def net_support(self) -> int:
        """Support count minus contradiction count."""
        return self.support_count - self.contradiction_count


# -- AssertionClaimLink -----------------------------------------------------


@dataclass
class AssertionClaimLink:
    """Links an assertion to a constituent evidence claim.

    Every claim that contributes to an assertion gets a link record.
    The link_type indicates whether the claim supports or contradicts
    the assertion. contribution_weight captures how much influence
    the claim has on the assertion's aggregate confidence.

    Attributes:
        assertion_id: The assertion being linked to.
        claim_id: The evidence claim.
        link_type: "support" or "contradiction".
        contribution_weight: Influence on aggregate confidence (0-1).
        metadata: Extensible metadata (e.g., why this weight).
    """

    assertion_id: str
    claim_id: str
    link_type: str
    contribution_weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if self.link_type not in VALID_LINK_TYPES:
            raise ValueError(
                f"Invalid link_type {self.link_type!r}. "
                f"Must be one of {sorted(VALID_LINK_TYPES)}"
            )
        if not 0.0 <= self.contribution_weight <= 1.0:
            raise ValueError(
                f"contribution_weight must be 0-1, got "
                f"{self.contribution_weight}"
            )
