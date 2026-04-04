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
from src.assertions.edges import (
    ConceptExposure,
    DerivedEdge,
    PathCacheEntry,
    build_path_cache,
    compute_exposures,
    derive_edges,
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
    "AssertionRepository",
    "ConceptExposure",
    "ConfidenceBreakdown",
    "DerivedEdge",
    "PathCacheEntry",
    "ResolvedAssertion",
    "aggregate_assertion",
    "build_path_cache",
    "compute_exposures",
    "derive_edges",
    "make_assertion_id",
]
