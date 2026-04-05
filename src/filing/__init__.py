"""Filing lane — SEC filing ingestion and parsing.

Provides the FilingProvider interface, SEC policy, and output schemas
for the filing lane. Provider implementations (edgartools, SEC API
fallback) are in separate modules.
"""

from src.filing.adoption import (
    AdoptionBreakdown,
    FactInput,
    FactSignal,
    FilingAdoptionScore,
    SectionInput,
    SectionSignal,
    compute_filing_adoption,
)
from src.filing.config import FilingConfig
from src.filing.divergence import (
    DivergenceAlert,
    DivergenceReason,
    check_divergence,
)
from src.filing.drift import (
    DimensionDrift,
    DriftDecomposition,
    SectionChange,
    classify_by_dimension,
    compute_dimension_magnitude,
    compute_drift_decomposition,
    extract_section_changes,
)
from src.filing.edgartools_provider import EdgarToolsProvider
from src.filing.publisher import (
    AdoptionPayload,
    DivergencePayload,
    FilingPublicationResult,
    IssuerDivergenceSummary,
    build_adoption_payload,
    build_divergence_payload,
    build_issuer_summaries,
    prepare_filing_publication,
)
from src.filing.provider import FilingProvider, SECRateLimiter
from src.filing.sec_api_provider import SecApiProvider
from src.filing.schemas import (
    VALID_FILING_STATUSES,
    VALID_FILING_TYPES,
    FilingIdentity,
    FilingResult,
    FilingSection,
)
from src.filing.sec_policy import SECPolicy
from src.filing.persistence import (
    FilingAttachmentRecord,
    FilingRecord,
    FilingRepository,
    FilingSectionRecord,
    XBRLFactRecord,
    filing_result_to_records,
)

__all__ = [
    "AdoptionBreakdown",
    "AdoptionPayload",
    "DimensionDrift",
    "DivergenceAlert",
    "DivergencePayload",
    "DivergenceReason",
    "DriftDecomposition",
    "FilingPublicationResult",
    "IssuerDivergenceSummary",
    "VALID_FILING_STATUSES",
    "VALID_FILING_TYPES",
    "EdgarToolsProvider",
    "FactInput",
    "FactSignal",
    "FilingAdoptionScore",
    "FilingAttachmentRecord",
    "FilingConfig",
    "FilingIdentity",
    "FilingProvider",
    "FilingRecord",
    "FilingRepository",
    "FilingResult",
    "FilingSection",
    "FilingSectionRecord",
    "SECPolicy",
    "SECRateLimiter",
    "SectionChange",
    "SectionInput",
    "SectionSignal",
    "SecApiProvider",
    "XBRLFactRecord",
    "build_adoption_payload",
    "build_divergence_payload",
    "build_issuer_summaries",
    "check_divergence",
    "classify_by_dimension",
    "compute_dimension_magnitude",
    "compute_drift_decomposition",
    "compute_filing_adoption",
    "extract_section_changes",
    "filing_result_to_records",
    "prepare_filing_publication",
]
