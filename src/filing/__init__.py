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
from src.filing.edgartools_provider import EdgarToolsProvider
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
    "SectionInput",
    "SectionSignal",
    "SecApiProvider",
    "XBRLFactRecord",
    "compute_filing_adoption",
    "filing_result_to_records",
]
