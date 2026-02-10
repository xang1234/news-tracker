"""
Dependency injection for FastAPI endpoints.
"""

from typing import AsyncGenerator

import redis.asyncio as redis

from src.alerts.broadcaster import AlertBroadcaster
from src.alerts.repository import AlertRepository
from src.feedback.repository import FeedbackRepository
from src.config.settings import get_settings
from src.graph.causal_graph import CausalGraph
from src.graph.config import GraphConfig
from src.graph.propagation import SentimentPropagation
from src.graph.storage import GraphRepository
from src.embedding.config import EmbeddingConfig
from src.embedding.service import EmbeddingService
from src.sentiment.aggregation import SentimentAggregator
from src.sentiment.config import SentimentConfig
from src.sentiment.service import SentimentService
from src.storage.database import Database
from src.storage.repository import DocumentRepository
from src.themes.ranking import ThemeRankingService
from src.themes.repository import ThemeRepository
from src.vectorstore.manager import VectorStoreManager
from src.ner.config import NERConfig
from src.ner.service import NERService
from src.keywords.config import KeywordsConfig
from src.keywords.service import KeywordsService
from src.event_extraction.config import EventExtractionConfig
from src.event_extraction.patterns import PatternExtractor
from src.security_master.repository import SecurityMasterRepository

# Global service instances (initialized on first request)
_embedding_service: EmbeddingService | None = None
_sentiment_service: SentimentService | None = None
_redis_client: redis.Redis | None = None
_vector_store_manager: VectorStoreManager | None = None
_database: Database | None = None
_theme_repository: ThemeRepository | None = None
_document_repository: DocumentRepository | None = None
_sentiment_aggregator: SentimentAggregator | None = None
_ranking_service: ThemeRankingService | None = None
_alert_repository: AlertRepository | None = None
_causal_graph: CausalGraph | None = None
_propagation_service: SentimentPropagation | None = None
_feedback_repository: FeedbackRepository | None = None
_alert_broadcaster: AlertBroadcaster | None = None
_graph_repository: GraphRepository | None = None
_ner_service: NERService | None = None
_keywords_service: KeywordsService | None = None
_pattern_extractor: PatternExtractor | None = None
_security_master_repository: SecurityMasterRepository | None = None


async def get_database() -> Database:
    """Get database instance (singleton)."""
    global _database

    if _database is None:
        _database = Database()
        await _database.connect()

    return _database


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


async def get_theme_repository() -> ThemeRepository:
    """
    Get theme repository instance.

    Creates a singleton repository backed by the shared Database.
    """
    global _theme_repository, _database

    if _theme_repository is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _theme_repository = ThemeRepository(_database)

    return _theme_repository


async def get_document_repository() -> DocumentRepository:
    """
    Get document repository instance.

    Creates a singleton repository backed by the shared Database.
    """
    global _document_repository, _database

    if _document_repository is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _document_repository = DocumentRepository(_database)

    return _document_repository


async def get_ranking_service() -> ThemeRankingService:
    """
    Get ranking service instance.

    Creates a singleton service backed by the shared ThemeRepository.
    """
    global _ranking_service, _theme_repository, _database

    if _ranking_service is None:
        if _theme_repository is None:
            if _database is None:
                _database = Database()
                await _database.connect()
            _theme_repository = ThemeRepository(_database)

        _ranking_service = ThemeRankingService(theme_repo=_theme_repository)

    return _ranking_service


async def get_alert_repository() -> AlertRepository:
    """
    Get alert repository instance.

    Creates a singleton repository backed by the shared Database.
    """
    global _alert_repository, _database

    if _alert_repository is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _alert_repository = AlertRepository(_database)

    return _alert_repository


async def get_feedback_repository() -> FeedbackRepository:
    """
    Get feedback repository instance.

    Creates a singleton repository backed by the shared Database.
    """
    global _feedback_repository, _database

    if _feedback_repository is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _feedback_repository = FeedbackRepository(_database)

    return _feedback_repository


def get_sentiment_aggregator() -> SentimentAggregator:
    """
    Get sentiment aggregator instance.

    Stateless aggregator — no DB or async needed.
    """
    global _sentiment_aggregator

    if _sentiment_aggregator is None:
        _sentiment_aggregator = SentimentAggregator()

    return _sentiment_aggregator


async def get_graph_repository() -> GraphRepository:
    """Get graph repository instance (singleton)."""
    global _graph_repository, _database

    if _graph_repository is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _graph_repository = GraphRepository(_database)

    return _graph_repository


async def get_causal_graph() -> CausalGraph:
    """
    Get causal graph instance.

    Creates a singleton CausalGraph backed by the shared Database.
    """
    global _causal_graph, _database

    if _causal_graph is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _causal_graph = CausalGraph(_database, config=GraphConfig())

    return _causal_graph


async def get_propagation_service() -> SentimentPropagation:
    """
    Get sentiment propagation service instance.

    Creates a singleton service backed by the shared CausalGraph.
    """
    global _propagation_service, _causal_graph, _database

    if _propagation_service is None:
        if _causal_graph is None:
            if _database is None:
                _database = Database()
                await _database.connect()
            _causal_graph = CausalGraph(_database, config=GraphConfig())

        _propagation_service = SentimentPropagation(
            graph=_causal_graph,
            config=GraphConfig(),
        )

    return _propagation_service


async def get_alert_broadcaster() -> AlertBroadcaster:
    """Get or create the alert broadcaster singleton.

    Initializes the broadcaster, connects to Redis, and starts the
    background subscriber task. Called during app lifespan startup.
    """
    global _alert_broadcaster, _redis_client

    if _alert_broadcaster is None:
        settings = get_settings()

        if _redis_client is None:
            _redis_client = redis.from_url(
                str(settings.redis_url),
                encoding="utf-8",
                decode_responses=True,
            )

        _alert_broadcaster = AlertBroadcaster(
            max_connections=settings.ws_alerts_max_connections,
            heartbeat_interval=settings.ws_alerts_heartbeat_seconds,
        )
        await _alert_broadcaster.start(_redis_client)

    return _alert_broadcaster


async def stop_alert_broadcaster() -> None:
    """Stop the alert broadcaster (called during shutdown)."""
    global _alert_broadcaster

    if _alert_broadcaster is not None:
        await _alert_broadcaster.stop()
        _alert_broadcaster = None


def get_ner_service() -> NERService:
    """Get NER service instance (CPU-only, no Redis needed)."""
    global _ner_service

    if _ner_service is None:
        _ner_service = NERService(config=NERConfig())

    return _ner_service


def get_keywords_service() -> KeywordsService:
    """Get keywords service instance (CPU-only, no Redis needed)."""
    global _keywords_service

    if _keywords_service is None:
        _keywords_service = KeywordsService(config=KeywordsConfig())

    return _keywords_service


def get_pattern_extractor() -> PatternExtractor:
    """Get pattern extractor instance (CPU-only, no Redis needed)."""
    global _pattern_extractor

    if _pattern_extractor is None:
        _pattern_extractor = PatternExtractor(config=EventExtractionConfig())

    return _pattern_extractor


async def get_security_master_repository() -> SecurityMasterRepository:
    """Get security master repository instance (singleton)."""
    global _security_master_repository, _database

    if _security_master_repository is None:
        if _database is None:
            _database = Database()
            await _database.connect()

        _security_master_repository = SecurityMasterRepository(_database)

    return _security_master_repository


async def cleanup_dependencies() -> None:
    """Clean up global dependencies on shutdown."""
    global _embedding_service, _sentiment_service, _redis_client, _vector_store_manager, _database
    global _theme_repository, _document_repository, _sentiment_aggregator, _ranking_service
    global _alert_repository, _feedback_repository, _causal_graph, _propagation_service, _alert_broadcaster
    global _graph_repository, _ner_service, _keywords_service, _pattern_extractor, _security_master_repository

    _vector_store_manager = None
    _theme_repository = None
    _ner_service = None
    _keywords_service = None
    _pattern_extractor = None
    _document_repository = None
    _sentiment_aggregator = None
    _ranking_service = None
    _alert_repository = None
    _feedback_repository = None
    _causal_graph = None
    _propagation_service = None
    _graph_repository = None
    _security_master_repository = None

    if _alert_broadcaster is not None:
        await _alert_broadcaster.stop()
        _alert_broadcaster = None

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
