"""
Embedding generation module for financial documents.

This module provides:
- EmbeddingService: Core service for generating FinBERT and MiniLM embeddings
- EmbeddingQueue: Redis stream wrapper for embedding job queue
- EmbeddingJob: Data class representing an embedding job
- EmbeddingWorker: Worker that consumes queue and updates documents
- EmbeddingConfig: Configuration settings for the embedding service
- ModelType: Enum for selecting embedding model (FINBERT, MINILM)
"""

from src.embedding.config import EmbeddingConfig
from src.embedding.queue import EmbeddingJob, EmbeddingQueue
from src.embedding.service import EmbeddingService, ModelType, get_embedding_service
from src.embedding.worker import EmbeddingWorker

__all__ = [
    "EmbeddingConfig",
    "EmbeddingJob",
    "EmbeddingQueue",
    "EmbeddingService",
    "EmbeddingWorker",
    "ModelType",
    "get_embedding_service",
]
