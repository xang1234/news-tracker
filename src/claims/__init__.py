"""Evidence claims — the atomic intelligence unit.

Every assertion, score, graph edge, and published artifact
traces back to evidence claims with deterministic keys,
source lineage, and bitemporal validity.
"""

from src.claims.config import ClaimsConfig
from src.claims.llm_gate import (
    DenyReason,
    FallbackGate,
    FallbackProvenance,
    GateDecision,
    GateVerdict,
)
from src.claims.repository import ClaimRepository
from src.claims.resolver import EntityResolver, ResolverResult, ResolverTier
from src.claims.review import (
    VALID_TASK_STATUSES,
    VALID_TASK_TYPES,
    VALID_TRIGGER_REASONS,
    ReviewTask,
    make_review_task_id,
    validate_review_transition,
)
from src.claims.review_repository import ReviewRepository
from src.claims.schemas import (
    VALID_CLAIM_STATUSES,
    VALID_EXTRACTION_METHODS,
    VALID_SOURCE_TYPES,
    EvidenceClaim,
    make_claim_id,
    make_claim_key,
)

__all__ = [
    "VALID_CLAIM_STATUSES",
    "VALID_EXTRACTION_METHODS",
    "VALID_SOURCE_TYPES",
    "VALID_TASK_STATUSES",
    "VALID_TASK_TYPES",
    "VALID_TRIGGER_REASONS",
    "ClaimRepository",
    "ClaimsConfig",
    "DenyReason",
    "EntityResolver",
    "EvidenceClaim",
    "FallbackGate",
    "FallbackProvenance",
    "GateDecision",
    "GateVerdict",
    "ResolverResult",
    "ResolverTier",
    "ReviewRepository",
    "ReviewTask",
    "make_claim_id",
    "make_claim_key",
    "make_review_task_id",
    "validate_review_transition",
]
