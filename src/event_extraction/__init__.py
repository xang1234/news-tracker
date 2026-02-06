"""
Event extraction for financial text.

This module provides pattern-based event extraction from semiconductor
financial text, following the established NER/Keywords opt-in service pattern
with lazy initialization and graceful error handling.

Components:
- EventExtractionConfig: Configuration for the extraction service
- EventRecord: Dataclass representing a structured event
- EventType: Literal type for event categories
- PatternExtractor: Regex-based event extractor
- TimeNormalizer: Normalizer for temporal references
"""

from src.event_extraction.config import EventExtractionConfig
from src.event_extraction.normalizer import TimeNormalizer
from src.event_extraction.patterns import PatternExtractor
from src.event_extraction.schemas import EventRecord, EventType

__all__ = [
    "EventExtractionConfig",
    "EventRecord",
    "EventType",
    "PatternExtractor",
    "TimeNormalizer",
]
