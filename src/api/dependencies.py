"""
Dependency injection for FastAPI endpoints.
"""

from typing import AsyncGenerator

import redis.asyncio as redis

from src.config.settings import get_settings
from src.embedding.config import EmbeddingConfig
from src.embedding.service import EmbeddingService
from src.sentiment.config import SentimentConfig
from src.sentiment.service import SentimentService
from src.storage.database import Database
from src.vectorstore.manager import VectorStoreManager

# Global service instances (initialized on first request)
_embedding_service: EmbeddingService | None = None
_sentiment_service: SentimentService | None = None
_redis_client: redis.Redis | None = None
_vector_store_manager: VectorStoreManager | None = None
_database: Database | None = None


async def get_redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Get Redis client for caching."""
    global _redis_client

    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
        )

    yield _redis_client


async def get_embedding_service() -> EmbeddingService:
    """
    Get embedding service instance.

    Creates a singleton service with Redis caching enabled.
    """
    global _embedding_service, _redis_client

    if _embedding_service is None:
        settings = get_settings()

        # Initialize Redis client if not already done
        if _redis_client is None:
            _redis_client = redis.from_url(
                str(settings.redis_url),
                encoding="utf-8",
                decode_responses=True,
            )

        # Create embedding config from settings
        config = EmbeddingConfig(
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            use_fp16=settings.embedding_use_fp16,
            device=settings.embedding_device,
            cache_enabled=settings.embedding_cache_enabled,
            cache_ttl_hours=settings.embedding_cache_ttl_hours,
        )

        _embedding_service = EmbeddingService(
            config=config,
            redis_client=_redis_client,
        )

    return _embedding_service


async def get_sentiment_service() -> SentimentService:
    """
    Get sentiment service instance.

    Creates a singleton service with Redis caching enabled.
    """
    global _sentiment_service, _redis_client

    if _sentiment_service is None:
        settings = get_settings()

        # Initialize Redis client if not already done
        if _redis_client is None:
            _redis_client = redis.from_url(
                str(settings.redis_url),
                encoding="utf-8",
                decode_responses=True,
            )

        # Create sentiment config from settings
        config = SentimentConfig(
            model_name=settings.sentiment_model_name,
            batch_size=settings.sentiment_batch_size,
            use_fp16=settings.sentiment_use_fp16,
            device=settings.sentiment_device,
            stream_name=settings.sentiment_stream_name,
            consumer_group=settings.sentiment_consumer_group,
            cache_enabled=settings.sentiment_cache_enabled,
            cache_ttl_hours=settings.sentiment_cache_ttl_hours,
            enable_entity_sentiment=settings.sentiment_enable_entity_sentiment,
        )

        _sentiment_service = SentimentService(
            config=config,
            redis_client=_redis_client,
        )

    return _sentiment_service


async def get_vector_store_manager() -> "VectorStoreManager":
    """
    Get vector store manager instance.

    Creates a singleton manager with Database, PgVectorStore, and EmbeddingService.
    """
    global _vector_store_manager, _database, _embedding_service, _redis_client

    if _vector_store_manager is None:
        from src.storage.repository import DocumentRepository
        from src.vectorstore.config import VectorStoreConfig
        from src.vectorstore.pgvector_store import PgVectorStore

        settings = get_settings()

        # Initialize database if needed
        if _database is None:
            _database = Database()
            await _database.connect()

        # Initialize embedding service if needed
        if _embedding_service is None:
            if _redis_client is None:
                _redis_client = redis.from_url(
                    str(settings.redis_url),
                    encoding="utf-8",
                    decode_responses=True,
                )

            config = EmbeddingConfig(
                model_name=settings.embedding_model_name,
                batch_size=settings.embedding_batch_size,
                use_fp16=settings.embedding_use_fp16,
                device=settings.embedding_device,
                cache_enabled=settings.embedding_cache_enabled,
                cache_ttl_hours=settings.embedding_cache_ttl_hours,
            )
            _embedding_service = EmbeddingService(
                config=config,
                redis_client=_redis_client,
            )

        # Create vector store and manager
        repository = DocumentRepository(_database)
        vector_store_config = VectorStoreConfig(
            default_limit=settings.vectorstore_default_limit,
            default_threshold=settings.vectorstore_default_threshold,
            centroid_default_limit=settings.vectorstore_centroid_limit,
            centroid_default_threshold=settings.vectorstore_centroid_threshold,
        )
        vector_store = PgVectorStore(
            database=_database,
            repository=repository,
            config=vector_store_config,
        )
        _vector_store_manager = VectorStoreManager(
            vector_store=vector_store,
            embedding_service=_embedding_service,
            config=vector_store_config,
        )

    return _vector_store_manager


async def cleanup_dependencies() -> None:
    """Clean up global dependencies on shutdown."""
    global _embedding_service, _sentiment_service, _redis_client, _vector_store_manager, _database

    _vector_store_manager = None

    if _embedding_service is not None:
        await _embedding_service.close()
        _embedding_service = None

    if _sentiment_service is not None:
        await _sentiment_service.close()
        _sentiment_service = None

    if _database is not None:
        await _database.close()
        _database = None

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
