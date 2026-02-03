"""
Named Entity Recognition (NER) for financial text.

This module provides domain-specific entity extraction for semiconductor
and financial news content, complementing the simpler TickerExtractor
with richer entity types and coreference resolution.

Components:
- NERConfig: Configuration for the NER service
- FinancialEntity: Dataclass representing an extracted entity
- NERService: Main service for entity extraction
"""

from src.ner.config import NERConfig
from src.ner.schemas import EntityType, FinancialEntity
from src.ner.service import NERService

__all__ = [
    "NERConfig",
    "EntityType",
    "FinancialEntity",
    "NERService",
]
