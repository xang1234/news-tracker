"""
Abstract base class and data models for vector store implementations.

Defines the interface that all vector store backends must implement,
plus shared data structures for search results and filters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorSearchResult:
    """
    Result from a vector similarity search.

    Attributes:
        document_id: Unique identifier of the matched document
        score: Cosine similarity score (0.0-1.0, higher is more similar)
        metadata: Document metadata (platform, tickers, author info, etc.)
        embedding: Optional embedding vector (only included if requested)
    """

    document_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        """Validate score is in valid range."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


@dataclass
class VectorSearchFilter:
    """
    Filter criteria for vector searches.

    All filters are optional and combined with AND logic.
    Array filters (platforms, tickers, theme_ids) use ANY/overlap semantics.

    Attributes:
        platforms: Filter to documents from these platforms (OR within)
        tickers: Filter to documents mentioning these tickers (OR within)
        theme_ids: Filter to documents with these theme IDs (OR within)
        min_authority_score: Minimum authority score threshold
        exclude_ids: Document IDs to exclude from results
    """

    platforms: list[str] | None = None
    tickers: list[str] | None = None
    theme_ids: list[str] | None = None
    min_authority_score: float | None = None
    exclude_ids: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate filter values."""
        if self.min_authority_score is not None:
            if not 0.0 <= self.min_authority_score <= 1.0:
                raise ValueError(
                    f"min_authority_score must be between 0.0 and 1.0, "
                    f"got {self.min_authority_score}"
                )

    @property
    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return (
            self.platforms is None
            and self.tickers is None
            and self.theme_ids is None
            and self.min_authority_score is None
            and self.exclude_ids is None
        )


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    Defines the core interface for storing and searching embeddings.
    Implementations may wrap different backends (pgvector, Pinecone, Milvus, etc.)
    while providing a consistent API.

    All methods are async to support non-blocking I/O.
    """

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> int:
        """
        Insert or update embeddings with associated metadata.

        Args:
            ids: Document IDs
            embeddings: Embedding vectors (must match dimensionality)
            metadata: Optional metadata dicts for each document

        Returns:
            Number of documents successfully upserted
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        threshold: float = 0.7,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar documents using a query embedding.

        Args:
            query_embedding: Query vector to find similar documents
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0.0-1.0)
            filters: Optional filter criteria

        Returns:
            List of search results sorted by similarity (descending)
        """
        ...

    @abstractmethod
    async def search_by_centroid(
        self,
        centroid: list[float],
        limit: int = 100,
        threshold: float = 0.5,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for documents near a cluster centroid.

        Similar to search() but with relaxed default threshold,
        intended for finding documents belonging to a topic cluster.

        Args:
            centroid: Cluster centroid vector
            limit: Maximum number of results
            threshold: Minimum similarity (default lower for centroid search)
            filters: Optional filter criteria

        Returns:
            List of search results sorted by similarity
        """
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: Document IDs to delete

        Returns:
            Number of documents deleted
        """
        ...

    @abstractmethod
    async def get_by_ids(
        self,
        ids: list[str],
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: Document IDs to retrieve
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of results (score will be 1.0 for exact matches)
        """
        ...
