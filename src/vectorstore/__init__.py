"""
Vector store abstraction layer for semantic search.

This module provides a clean abstraction over the existing pgvector infrastructure,
enabling future flexibility for alternative vector databases while leveraging
the existing HNSW indexes and similarity_search methods.

Main components:
- VectorStore: Abstract base class defining the vector store interface
- PgVectorStore: pgvector implementation using DocumentRepository
- VectorStoreManager: High-level orchestration (embed + store + search)
- VectorSearchResult: Search result data class with score and metadata
- VectorSearchFilter: Filter criteria for searches
"""

from src.vectorstore.base import (
    VectorSearchFilter,
    VectorSearchResult,
    VectorStore,
)
from src.vectorstore.config import VectorStoreConfig
from src.vectorstore.manager import VectorStoreManager
from src.vectorstore.pgvector_store import PgVectorStore

__all__ = [
    "VectorStore",
    "VectorSearchResult",
    "VectorSearchFilter",
    "VectorStoreConfig",
    "PgVectorStore",
    "VectorStoreManager",
]
