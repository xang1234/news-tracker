"""Claim quality checks, dead-letter handling, and quarantine rules.

Quality checks run before a claim is persisted. They return a
QualityVerdict that tells the caller whether to accept, quarantine,
or dead-letter the claim. Every non-pass verdict preserves full
context so the failure can be diagnosed and replayed.

Check categories:
    - Structural: required fields, valid spans, sane sizes
    - Semantic: self-referencing, empty predicates, confidence floor
    - Safety: payload size limits, source text truncation detection

Dead-letter records capture unrecoverable failures with the original
source context. Quarantined claims are persisted but flagged for
review before they can become authoritative.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.claims.schemas import EvidenceClaim

logger = logging.getLogger(__name__)


# -- Verdict types ---------------------------------------------------------


class Disposition(str, Enum):
    """What should happen to a claim after quality checks."""

    ACCEPT = "accept"
    QUARANTINE = "quarantine"
    DEAD_LETTER = "dead_letter"


class CheckCode(str, Enum):
    """Specific quality check that flagged the claim."""

    EMPTY_SUBJECT = "empty_subject"
    EMPTY_PREDICATE = "empty_predicate"
    SELF_REFERENCING = "self_referencing"
    CONFIDENCE_BELOW_FLOOR = "confidence_below_floor"
    INVALID_SPAN = "invalid_span"
    SOURCE_TEXT_TOO_LONG = "source_text_too_long"
    METADATA_TOO_LARGE = "metadata_too_large"
    MISSING_SOURCE_ID = "missing_source_id"
    MISSING_CONTRACT_VERSION = "missing_contract_version"


@dataclass(frozen=True)
class QualityVerdict:
    """Result of running quality checks on a claim.

    Attributes:
        disposition: What should happen (accept, quarantine, dead_letter).
        checks_failed: List of check codes that failed.
        messages: Human-readable messages for each failure.
        claim_id: The claim that was checked (for tracing).
    """

    disposition: Disposition
    checks_failed: list[CheckCode] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    claim_id: str = ""

    @property
    def passed(self) -> bool:
        return self.disposition == Disposition.ACCEPT


# -- Quality check functions -----------------------------------------------

# Configurable thresholds
CONFIDENCE_FLOOR = 0.05
MAX_SOURCE_TEXT_LENGTH = 50_000
MAX_METADATA_SIZE = 100_000  # bytes when JSON-serialized


def run_quality_checks(
    claim: EvidenceClaim,
    *,
    confidence_floor: float = CONFIDENCE_FLOOR,
    max_source_text_length: int = MAX_SOURCE_TEXT_LENGTH,
    max_metadata_size: int = MAX_METADATA_SIZE,
) -> QualityVerdict:
    """Run all quality checks on a claim.

    Returns QualityVerdict with disposition and failure details.
    Dead-letter checks (structural failures) take priority over
    quarantine checks (semantic/safety concerns).
    """
    dead_letter_codes: list[CheckCode] = []
    dead_letter_msgs: list[str] = []
    quarantine_codes: list[CheckCode] = []
    quarantine_msgs: list[str] = []

    # -- Structural checks (dead-letter if failed) --

    if not claim.subject_text or not claim.subject_text.strip():
        dead_letter_codes.append(CheckCode.EMPTY_SUBJECT)
        dead_letter_msgs.append("Subject text is empty or whitespace-only")

    if not claim.predicate or not claim.predicate.strip():
        dead_letter_codes.append(CheckCode.EMPTY_PREDICATE)
        dead_letter_msgs.append("Predicate is empty or whitespace-only")

    if not claim.source_id:
        dead_letter_codes.append(CheckCode.MISSING_SOURCE_ID)
        dead_letter_msgs.append("Source ID is missing")

    if not claim.contract_version:
        dead_letter_codes.append(CheckCode.MISSING_CONTRACT_VERSION)
        dead_letter_msgs.append("Contract version is missing")

    if (
        claim.source_span_start is not None
        and claim.source_span_end is not None
        and claim.source_span_start > claim.source_span_end
    ):
        dead_letter_codes.append(CheckCode.INVALID_SPAN)
        dead_letter_msgs.append(
            f"Source span is inverted: start={claim.source_span_start} > "
            f"end={claim.source_span_end}"
        )

    if dead_letter_codes:
        return QualityVerdict(
            disposition=Disposition.DEAD_LETTER,
            checks_failed=dead_letter_codes,
            messages=dead_letter_msgs,
            claim_id=claim.claim_id,
        )

    # -- Semantic/safety checks (quarantine if failed) --

    if (
        claim.subject_concept_id
        and claim.object_concept_id
        and claim.subject_concept_id == claim.object_concept_id
    ):
        quarantine_codes.append(CheckCode.SELF_REFERENCING)
        quarantine_msgs.append(
            f"Subject and object resolve to the same concept: "
            f"{claim.subject_concept_id}"
        )

    if claim.confidence < confidence_floor:
        quarantine_codes.append(CheckCode.CONFIDENCE_BELOW_FLOOR)
        quarantine_msgs.append(
            f"Confidence {claim.confidence:.3f} is below floor "
            f"{confidence_floor:.3f}"
        )

    if (
        claim.source_text
        and len(claim.source_text) > max_source_text_length
    ):
        quarantine_codes.append(CheckCode.SOURCE_TEXT_TOO_LONG)
        quarantine_msgs.append(
            f"Source text length {len(claim.source_text)} exceeds "
            f"limit {max_source_text_length}"
        )

    metadata_bytes = len(json.dumps(claim.metadata).encode())
    if metadata_bytes > max_metadata_size:
        quarantine_codes.append(CheckCode.METADATA_TOO_LARGE)
        quarantine_msgs.append(
            f"Metadata size {metadata_bytes} bytes exceeds "
            f"limit {max_metadata_size}"
        )

    if quarantine_codes:
        return QualityVerdict(
            disposition=Disposition.QUARANTINE,
            checks_failed=quarantine_codes,
            messages=quarantine_msgs,
            claim_id=claim.claim_id,
        )

    return QualityVerdict(
        disposition=Disposition.ACCEPT,
        claim_id=claim.claim_id,
    )


# -- Dead-letter record ----------------------------------------------------


VALID_DL_REASONS = frozenset(
    {
        "quality_check_failed",
        "extraction_error",
        "parse_error",
        "validation_error",
        "timeout",
    }
)


def make_dead_letter_id(
    lane: str,
    run_id: str,
    source_id: str,
    error_hash: str,
) -> str:
    """Generate a deterministic dead-letter record ID."""
    parts = [lane, run_id, source_id, error_hash]
    key_input = "\x00".join(parts)
    return f"dl_{hashlib.sha256(key_input.encode()).hexdigest()[:16]}"


@dataclass
class DeadLetterRecord:
    """An unrecoverable extraction failure captured for replay.

    Attributes:
        record_id: Deterministic ID for dedup.
        lane: Which processing lane produced the failure.
        run_id: Lane run that failed.
        source_id: Source document/section that was being processed.
        reason: Category of failure.
        error_message: Human-readable error description.
        error_detail: Full error context (traceback, check codes, etc.).
        source_text: The raw text that was being extracted (for replay).
        claim_snapshot: Partial claim data if extraction got that far.
        metadata: Extensible metadata.
    """

    record_id: str
    lane: str
    run_id: str
    source_id: str
    reason: str
    error_message: str
    error_detail: dict[str, Any] = field(default_factory=dict)
    source_text: str | None = None
    claim_snapshot: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if self.reason not in VALID_DL_REASONS:
            raise ValueError(
                f"Invalid dead-letter reason {self.reason!r}. "
                f"Must be one of {sorted(VALID_DL_REASONS)}"
            )


def capture_dead_letter(
    *,
    lane: str,
    run_id: str,
    source_id: str,
    reason: str,
    error_message: str,
    error_detail: dict[str, Any] | None = None,
    source_text: str | None = None,
    claim: EvidenceClaim | None = None,
) -> DeadLetterRecord:
    """Build a dead-letter record from a failure context.

    If a partial claim exists, its fields are snapshot into
    claim_snapshot for later inspection. The source_text is
    preserved for replay.
    """
    error_hash = hashlib.sha256(
        error_message.encode()
    ).hexdigest()[:8]

    claim_snapshot = None
    if claim is not None:
        claim_snapshot = {
            "claim_id": claim.claim_id,
            "claim_key": claim.claim_key,
            "subject_text": claim.subject_text,
            "predicate": claim.predicate,
            "object_text": claim.object_text,
            "confidence": claim.confidence,
            "extraction_method": claim.extraction_method,
            "metadata": claim.metadata,
        }

    return DeadLetterRecord(
        record_id=make_dead_letter_id(lane, run_id, source_id, error_hash),
        lane=lane,
        run_id=run_id,
        source_id=source_id,
        reason=reason,
        error_message=error_message,
        error_detail=error_detail or {},
        source_text=source_text,
        claim_snapshot=claim_snapshot,
    )


def verdict_to_dead_letter(
    verdict: QualityVerdict,
    *,
    claim: EvidenceClaim,
    run_id: str,
    source_text: str | None = None,
) -> DeadLetterRecord:
    """Convert a failed QualityVerdict into a dead-letter record."""
    return capture_dead_letter(
        lane=claim.lane,
        run_id=run_id,
        source_id=claim.source_id,
        reason="quality_check_failed",
        error_message="; ".join(verdict.messages),
        error_detail={
            "checks_failed": [c.value for c in verdict.checks_failed],
            "disposition": verdict.disposition.value,
        },
        source_text=source_text or claim.source_text,
        claim=claim,
    )
