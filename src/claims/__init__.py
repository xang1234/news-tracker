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
    "make_claim_id",
    "make_claim_key",
]
