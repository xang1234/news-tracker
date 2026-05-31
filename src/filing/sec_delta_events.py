"""Facade exports for SEC filing-delta event computation."""

from src.filing.sec_delta_extractor import compute_sec_filing_delta_events
from src.filing.sec_delta_models import (
    SEC_FILING_DELTA_EVENT_TYPES,
    SECFactObservation,
    SECFilingDeltaEvent,
    make_sec_delta_event_id,
)
from src.filing.sec_delta_repository import SECFilingDeltaRepository

__all__ = [
    "SEC_FILING_DELTA_EVENT_TYPES",
    "SECFactObservation",
    "SECFilingDeltaEvent",
    "SECFilingDeltaRepository",
    "compute_sec_filing_delta_events",
    "make_sec_delta_event_id",
]
