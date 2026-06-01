"""Compatibility exports for research innovation primitives."""

from src.innovation.arxiv_provider import ArxivResearchProvider
from src.innovation.openalex import OpenAlexResearchProvider
from src.innovation.research_linking import (
    ResearchSignalLinker,
    deduplicate_research_records,
)
from src.innovation.research_schemas import (
    ResearchProviderError,
    ResearchProviderResponseError,
    ResearchQuery,
    ResearchRecord,
    ResearchSignal,
)

__all__ = [
    "ArxivResearchProvider",
    "OpenAlexResearchProvider",
    "ResearchProviderError",
    "ResearchProviderResponseError",
    "ResearchQuery",
    "ResearchRecord",
    "ResearchSignal",
    "ResearchSignalLinker",
    "deduplicate_research_records",
]
