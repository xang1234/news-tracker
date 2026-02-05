"""
Keyword extraction for financial text using TextRank algorithm.

This module provides keyword extraction using the rapid-textrank library,
following the established NER service pattern with opt-in activation,
lazy initialization, and graceful error handling.

Components:
- KeywordsConfig: Configuration for the keywords service
- ExtractedKeyword: Dataclass representing an extracted keyword
- KeywordsService: Main service for keyword extraction
"""

from src.keywords.config import KeywordsConfig
from src.keywords.schemas import ExtractedKeyword
from src.keywords.service import KeywordsService

__all__ = [
    "KeywordsConfig",
    "ExtractedKeyword",
    "KeywordsService",
]
