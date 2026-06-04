"""Facade exports for SEC filing-delta event computation."""

from src.filing.sec_delta_extractor import compute_sec_filing_delta_events
from src.filing.sec_delta_models import (
    SEC_FILING_DELTA_EVENT_TYPES,
    SECFactObservation,
    SECFilingDeltaEvent,
    make_sec_delta_event_id,
)
from src.filing.sec_delta_publication import (
    SEC_DELTA_REASON_CODES,
    SECDeltaPublicationPayload,
    SECFactEvidenceReference,
    build_sec_delta_payload,
    build_sec_fact_evidence,
    classify_sec_fact_evidence,
    sec_delta_reason_code,
)
from src.filing.sec_delta_repository import SECFilingDeltaRepository

__all__ = [
    "SEC_FILING_DELTA_EVENT_TYPES",
    "SEC_DELTA_REASON_CODES",
    "SECDeltaPublicationPayload",
    "SECFactObservation",
    "SECFactEvidenceReference",
    "SECFilingDeltaEvent",
    "SECFilingDeltaRepository",
    "build_sec_delta_payload",
    "build_sec_fact_evidence",
    "classify_sec_fact_evidence",
    "compute_sec_filing_delta_events",
    "make_sec_delta_event_id",
    "sec_delta_reason_code",
]
