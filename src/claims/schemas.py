"""Schema definitions for evidence claims.

The evidence claim is the atomic intelligence unit. Every assertion,
score, graph edge, and published artifact traces back to claims.

Claim key is deterministic: sha256(lane + source_id + subject + predicate + object).
Retries and replays produce the same key, so writes are idempotent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.contracts.intelligence.lanes import VALID_LANES

VALID_CLAIM_STATUSES = frozenset(
    {"active", "superseded", "retracted", "disputed"}
)

VALID_SOURCE_TYPES = frozenset(
    {"document", "filing_section", "graph_edge", "manual"}
)

VALID_EXTRACTION_METHODS = frozenset(
    {"rule", "llm", "hybrid", "manual"}
)


def make_claim_key(
    lane: str,
    source_id: str,
    subject_text: str,
    predicate: str,
    object_text: str | None = None,
) -> str:
    """Generate a deterministic claim key for idempotent writes.

    Same inputs always produce the same key. This is the deduplication
    mechanism — replaying extraction with identical inputs yields the
    same claim_key, so ON CONFLICT handles it.
    """
    parts = [
        lane,
        source_id,
        subject_text.lower().strip(),
        predicate.lower().strip(),
        (object_text or "").lower().strip(),
    ]
    key_input = "|".join(parts)
    return f"clk_{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"


def make_claim_id(claim_key: str) -> str:
    """Generate a claim ID from a claim key.

    The claim_id is the primary key; claim_key is the dedup key.
    They are related but distinct to allow future key evolution
    without PK changes.
    """
    return f"claim_{hashlib.sha256(claim_key.encode()).hexdigest()[:12]}"


@dataclass
class EvidenceClaim:
    """An atomic evidence claim extracted from a source.

    Attributes:
        claim_id: Primary key (derived from claim_key).
        claim_key: Deterministic deduplication key.
        lane: Which processing lane extracted this claim.
        run_id: Lane run that produced this claim.
        source_id: ID of the source document/section/edge.
        source_type: Kind of source (document, filing_section, etc.).
        source_span_start: Character offset start in source text.
        source_span_end: Character offset end in source text.
        source_text: Extracted text span (for audit/display).
        subject_text: The subject of the claim (e.g., "TSMC").
        subject_concept_id: Resolved canonical concept ID for subject.
        predicate: What the claim asserts (e.g., "supplies_to").
        object_text: The object of the claim (e.g., "NVIDIA").
        object_concept_id: Resolved canonical concept ID for object.
        confidence: Extraction confidence (0-1).
        extraction_method: How the claim was extracted (rule, llm, etc.).
        claim_valid_from: When the claimed fact became true.
        claim_valid_to: When the claimed fact ceased to be true.
        source_published_at: When the source was published.
        contract_version: Contract version governing this claim.
        status: Lifecycle state (active, superseded, retracted, disputed).
        metadata: Extensible metadata.
    """

    claim_id: str
    claim_key: str
    lane: str
    source_id: str
    predicate: str
    subject_text: str
    source_type: str = "document"
    run_id: str | None = None
    source_span_start: int | None = None
    source_span_end: int | None = None
    source_text: str | None = None
    subject_concept_id: str | None = None
    object_text: str | None = None
    object_concept_id: str | None = None
    confidence: float = 0.5
    extraction_method: str = "rule"
    claim_valid_from: datetime | None = None
    claim_valid_to: datetime | None = None
    source_published_at: datetime | None = None
    contract_version: str = ""
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.lane not in VALID_LANES:
            raise ValueError(
                f"Invalid lane {self.lane!r}. "
                f"Must be one of {sorted(VALID_LANES)}"
            )
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"Invalid source_type {self.source_type!r}. "
                f"Must be one of {sorted(VALID_SOURCE_TYPES)}"
            )
        if self.extraction_method not in VALID_EXTRACTION_METHODS:
            raise ValueError(
                f"Invalid extraction_method {self.extraction_method!r}. "
                f"Must be one of {sorted(VALID_EXTRACTION_METHODS)}"
            )
        if self.status not in VALID_CLAIM_STATUSES:
            raise ValueError(
                f"Invalid claim status {self.status!r}. "
                f"Must be one of {sorted(VALID_CLAIM_STATUSES)}"
            )
