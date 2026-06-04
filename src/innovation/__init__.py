"""Innovation datasource primitives."""

from src.innovation.patents import (
    MissingPatentProviderCredentialError,
    PatentProviderError,
    PatentProviderResponseError,
    PatentQuery,
    PatentRecord,
    PatentSignal,
    PatentSignalLinker,
    PatentsViewProvider,
    StalePatentSnapshotError,
    deduplicate_patent_families,
    load_patentsview_bulk_snapshot,
)
from src.innovation.publish import (
    build_innovation_evidence_payload,
    group_innovation_evidence_by_concept,
    group_innovation_evidence_by_theme,
)
from src.innovation.repository import PatentSignalRepository
from src.innovation.research import (
    ArxivResearchProvider,
    OpenAlexResearchProvider,
    ResearchProviderError,
    ResearchProviderResponseError,
    ResearchQuery,
    ResearchRecord,
    ResearchSignal,
    ResearchSignalLinker,
    deduplicate_research_records,
)
from src.innovation.research_repository import ResearchSignalRepository

__all__ = [
    "ArxivResearchProvider",
    "MissingPatentProviderCredentialError",
    "OpenAlexResearchProvider",
    "PatentProviderError",
    "PatentProviderResponseError",
    "PatentQuery",
    "PatentRecord",
    "PatentSignal",
    "PatentSignalLinker",
    "PatentSignalRepository",
    "PatentsViewProvider",
    "ResearchProviderError",
    "ResearchProviderResponseError",
    "ResearchQuery",
    "ResearchRecord",
    "ResearchSignal",
    "ResearchSignalLinker",
    "ResearchSignalRepository",
    "StalePatentSnapshotError",
    "build_innovation_evidence_payload",
    "deduplicate_patent_families",
    "deduplicate_research_records",
    "group_innovation_evidence_by_concept",
    "group_innovation_evidence_by_theme",
    "load_patentsview_bulk_snapshot",
]
