"""Facade exports for SEC structured payload ingestion."""

from src.filing.sec_structured_models import (
    SECStructuredDataError,
    SECStructuredPayloadRecord,
)
from src.filing.sec_structured_provider import (
    SECStructuredDataProvider,
    SECStructuredHTTPClient,
    SECStructuredRepository,
)
from src.filing.sec_structured_repository import SECStructuredDataRepository

__all__ = [
    "SECStructuredDataError",
    "SECStructuredDataProvider",
    "SECStructuredDataRepository",
    "SECStructuredHTTPClient",
    "SECStructuredPayloadRecord",
    "SECStructuredRepository",
]
