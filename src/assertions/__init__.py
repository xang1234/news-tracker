"""Resolved assertions — stable current-belief objects.

Assertions aggregate raw evidence claims into durable belief objects
that downstream consumers (graph, scoring, publishing) can read
without touching raw claims directly. Every assertion retains
explicit support and contradiction links for auditability.
"""

from src.assertions.aggregation import (
    ConfidenceBreakdown,
    aggregate_assertion,
)
from src.assertions.claim_reconciliation import resolve_claim_subject
from src.assertions.edges import (
    ConceptExposure,
    DerivedEdge,
    PathCacheEntry,
    build_path_cache,
    compute_exposures,
    derive_edges,
)
from src.assertions.numeric_contradiction import (
    classify_numeric_links,
    numeric_link_types,
)
from src.assertions.numeric_reconciler import NumericReconciler
from src.assertions.predicate_contradiction import (
    antonym_of,
    classify_polarity_links,
    validity_overlaps,
)
from src.assertions.predicate_reconciler import PredicateContradictionReconciler
from src.assertions.recompute import (
    AssertionDelta,
    RecomputeResult,
    build_recompute_result,
    find_affected_assertion_ids,
    recompute_assertion,
)
from src.assertions.repository import AssertionRepository
from src.assertions.schemas import (
    VALID_ASSERTION_STATUSES,
    VALID_LINK_TYPES,
    AssertionClaimLink,
    ResolvedAssertion,
    make_assertion_id,
)

__all__ = [
    "VALID_ASSERTION_STATUSES",
    "VALID_LINK_TYPES",
    "AssertionClaimLink",
    "AssertionDelta",
    "AssertionRepository",
    "ConceptExposure",
    "ConfidenceBreakdown",
    "DerivedEdge",
    "NumericReconciler",
    "PathCacheEntry",
    "PredicateContradictionReconciler",
    "RecomputeResult",
    "ResolvedAssertion",
    "aggregate_assertion",
    "antonym_of",
    "build_path_cache",
    "build_recompute_result",
    "classify_numeric_links",
    "classify_polarity_links",
    "compute_exposures",
    "derive_edges",
    "find_affected_assertion_ids",
    "make_assertion_id",
    "numeric_link_types",
    "recompute_assertion",
    "resolve_claim_subject",
    "validity_overlaps",
]
