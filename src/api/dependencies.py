"""FastAPI dependencies backed by app-scoped services."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Literal, cast

import redis.asyncio as redis
from fastapi import FastAPI, Request

from src.alerts.broadcaster import AlertBroadcaster
from src.alerts.repository import AlertRepository
from src.assertions.repository import AssertionRepository
from src.claims.repository import ClaimRepository
from src.config.settings import get_settings
from src.embedding.config import EmbeddingConfig
from src.embedding.service import EmbeddingService
from src.event_extraction.config import EventExtractionConfig
from src.event_extraction.patterns import PatternExtractor
from src.feedback.repository import FeedbackRepository
from src.graph.causal_graph import CausalGraph
from src.graph.config import GraphConfig
from src.graph.propagation import SentimentPropagation
from src.graph.storage import GraphRepository
from src.keywords.config import KeywordsConfig
from src.keywords.service import KeywordsService
from src.narrative.repository import NarrativeRepository
from src.ner.config import NERConfig
from src.ner.service import NERService
from src.publish.repository import PublishRepository
from src.publish.read_model_repository import ReadModelRepository
from src.publish.service import PublishService
from src.security_master.repository import SecurityMasterRepository
from src.sentiment.aggregation import SentimentAggregator
from src.sentiment.config import SentimentConfig
from src.sentiment.service import SentimentService
from src.sources.repository import SourcesRepository
from src.storage.database import Database
from src.storage.repository import DocumentRepository
from src.themes.ranking import ThemeRankingService
from src.themes.repository import ThemeRepository
from src.vectorstore.config import VectorStoreConfig
from src.vectorstore.manager import VectorStoreManager
from src.vectorstore.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)


class AppServices:
    """App-scoped service container with lazy initialization."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._init_lock = asyncio.Lock()
        self._init_lock_sync = threading.Lock()

        self.database: Database | None = None
        self.redis_client: redis.Redis | None = None
        self.embedding_service: EmbeddingService | None = None
        self.sentiment_service: SentimentService | None = None
        self.vector_store_manager: VectorStoreManager | None = None
        self.theme_repository: ThemeRepository | None = None
        self.document_repository: DocumentRepository | None = None
        self.sentiment_aggregator: SentimentAggregator | None = None
        self.ranking_service: ThemeRankingService | None = None
        self.alert_repository: AlertRepository | None = None
        self.narrative_repository: NarrativeRepository | None = None
        self.causal_graph: CausalGraph | None = None
        self.propagation_service: SentimentPropagation | None = None
        self.feedback_repository: FeedbackRepository | None = None
        self.alert_broadcaster: AlertBroadcaster | None = None
        self.graph_repository: GraphRepository | None = None
        self.ner_service: NERService | None = None
        self.keywords_service: KeywordsService | None = None
        self.pattern_extractor: PatternExtractor | None = None
        self.security_master_repository: SecurityMasterRepository | None = None
        self.sources_repository: SourcesRepository | None = None
        self.publish_service: PublishService | None = None
        self.assertion_repository: AssertionRepository | None = None
        self.claim_repository: ClaimRepository | None = None

    async def get_database(self) -> Database:
        if self.database is None:
            async with self._init_lock:
                if self.database is None:
                    database = Database()
                    await database.connect()
                    self.database = database
        return self.database

    async def get_redis_client(self) -> redis.Redis:
        if self.redis_client is None:
            async with self._init_lock:
                if self.redis_client is None:
                    self.redis_client = cast(
                        redis.Redis,
                        redis.from_url(
                            str(self._settings.redis_url),
                            encoding="utf-8",
                            decode_responses=True,
                        ),
                    )
        return self.redis_client

    async def get_embedding_service(self) -> EmbeddingService:
        if self.embedding_service is None:
            redis_client = await self.get_redis_client()
            async with self._init_lock:
                if self.embedding_service is None:
                    self.embedding_service = EmbeddingService(
                        config=EmbeddingConfig(
                            model_name=self._settings.embedding_model_name,
                            batch_size=self._settings.embedding_batch_size,
                            use_fp16=self._settings.embedding_use_fp16,
                            backend=cast(
                                Literal["auto", "torch", "onnx"],
                                self._settings.embedding_backend,
                            ),
                            device=cast(
                                Literal["auto", "cpu", "cuda", "mps"],
                                self._settings.embedding_device,
                            ),
                            execution_provider=cast(
                                Literal["auto", "cpu", "cuda", "coreml"],
                                self._settings.embedding_execution_provider,
                            ),
                            onnx_model_path=self._settings.embedding_onnx_model_path,
                            onnx_minilm_model_path=self._settings.embedding_minilm_onnx_model_path,
                            cache_enabled=self._settings.embedding_cache_enabled,
                            cache_ttl_hours=self._settings.embedding_cache_ttl_hours,
                        ),
                        redis_client=redis_client,
                    )
        return self.embedding_service

    async def get_sentiment_service(self) -> SentimentService:
        if self.sentiment_service is None:
            redis_client = await self.get_redis_client()
            async with self._init_lock:
                if self.sentiment_service is None:
                    self.sentiment_service = SentimentService(
                        config=SentimentConfig(
                            model_name=self._settings.sentiment_model_name,
                            batch_size=self._settings.sentiment_batch_size,
                            use_fp16=self._settings.sentiment_use_fp16,
                            backend=cast(
                                Literal["auto", "torch", "onnx"],
                                self._settings.sentiment_backend,
                            ),
                            device=cast(
                                Literal["auto", "cpu", "cuda", "mps"],
                                self._settings.sentiment_device,
                            ),
                            execution_provider=cast(
                                Literal["auto", "cpu", "cuda", "coreml"],
                                self._settings.sentiment_execution_provider,
                            ),
                            onnx_model_path=self._settings.sentiment_onnx_model_path,
                            stream_name=self._settings.sentiment_stream_name,
                            consumer_group=self._settings.sentiment_consumer_group,
                            cache_enabled=self._settings.sentiment_cache_enabled,
                            cache_ttl_hours=self._settings.sentiment_cache_ttl_hours,
                            enable_entity_sentiment=self._settings.sentiment_enable_entity_sentiment,
                        ),
                        redis_client=redis_client,
                    )
        return self.sentiment_service

    async def get_vector_store_manager(self) -> VectorStoreManager:
        if self.vector_store_manager is None:
            database = await self.get_database()
            embedding_service = await self.get_embedding_service()
            async with self._init_lock:
                if self.vector_store_manager is None:
                    repository = DocumentRepository(database)
                    config = VectorStoreConfig(
                        default_limit=self._settings.vectorstore_default_limit,
                        default_threshold=self._settings.vectorstore_default_threshold,
                        centroid_default_limit=self._settings.vectorstore_centroid_limit,
                        centroid_default_threshold=self._settings.vectorstore_centroid_threshold,
                    )
                    vector_store = PgVectorStore(
                        database=database,
                        repository=repository,
                        config=config,
                    )
                    self.vector_store_manager = VectorStoreManager(
                        vector_store=vector_store,
                        embedding_service=embedding_service,
                        config=config,
                    )
        return self.vector_store_manager

    async def get_theme_repository(self) -> ThemeRepository:
        if self.theme_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.theme_repository is None:
                    self.theme_repository = ThemeRepository(database)
        return self.theme_repository

    async def get_document_repository(self) -> DocumentRepository:
        if self.document_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.document_repository is None:
                    self.document_repository = DocumentRepository(database)
        return self.document_repository

    async def get_ranking_service(self) -> ThemeRankingService:
        if self.ranking_service is None:
            theme_repository = await self.get_theme_repository()
            async with self._init_lock:
                if self.ranking_service is None:
                    self.ranking_service = ThemeRankingService(theme_repo=theme_repository)
        return self.ranking_service

    async def get_alert_repository(self) -> AlertRepository:
        if self.alert_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.alert_repository is None:
                    self.alert_repository = AlertRepository(database)
        return self.alert_repository

    async def get_narrative_repository(self) -> NarrativeRepository:
        if self.narrative_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.narrative_repository is None:
                    self.narrative_repository = NarrativeRepository(database)
        return self.narrative_repository

    async def get_feedback_repository(self) -> FeedbackRepository:
        if self.feedback_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.feedback_repository is None:
                    self.feedback_repository = FeedbackRepository(database)
        return self.feedback_repository

    def get_sentiment_aggregator(self) -> SentimentAggregator:
        if self.sentiment_aggregator is None:
            with self._init_lock_sync:
                if self.sentiment_aggregator is None:
                    self.sentiment_aggregator = SentimentAggregator()
        return self.sentiment_aggregator

    async def get_graph_repository(self) -> GraphRepository:
        if self.graph_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.graph_repository is None:
                    self.graph_repository = GraphRepository(database)
        return self.graph_repository

    async def get_causal_graph(self) -> CausalGraph:
        if self.causal_graph is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.causal_graph is None:
                    self.causal_graph = CausalGraph(database, config=GraphConfig())
        return self.causal_graph

    async def get_propagation_service(self) -> SentimentPropagation:
        if self.propagation_service is None:
            causal_graph = await self.get_causal_graph()
            async with self._init_lock:
                if self.propagation_service is None:
                    self.propagation_service = SentimentPropagation(
                        graph=causal_graph,
                        config=GraphConfig(),
                    )
        return self.propagation_service

    async def get_alert_broadcaster(self) -> AlertBroadcaster:
        if self.alert_broadcaster is None:
            redis_client = await self.get_redis_client()
            async with self._init_lock:
                if self.alert_broadcaster is None:
                    broadcaster = AlertBroadcaster(
                        max_connections=self._settings.ws_alerts_max_connections,
                        heartbeat_interval=self._settings.ws_alerts_heartbeat_seconds,
                    )
                    await broadcaster.start(redis_client)
                    self.alert_broadcaster = broadcaster
        return self.alert_broadcaster

    def get_ner_service(self) -> NERService:
        if self.ner_service is None:
            with self._init_lock_sync:
                if self.ner_service is None:
                    self.ner_service = NERService(config=NERConfig())
        return self.ner_service

    def get_keywords_service(self) -> KeywordsService:
        if self.keywords_service is None:
            with self._init_lock_sync:
                if self.keywords_service is None:
                    self.keywords_service = KeywordsService(config=KeywordsConfig())
        return self.keywords_service

    def get_pattern_extractor(self) -> PatternExtractor:
        if self.pattern_extractor is None:
            with self._init_lock_sync:
                if self.pattern_extractor is None:
                    self.pattern_extractor = PatternExtractor(config=EventExtractionConfig())
        return self.pattern_extractor

    async def get_security_master_repository(self) -> SecurityMasterRepository:
        if self.security_master_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.security_master_repository is None:
                    self.security_master_repository = SecurityMasterRepository(database)
        return self.security_master_repository

    async def get_sources_repository(self) -> SourcesRepository:
        if self.sources_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.sources_repository is None:
                    self.sources_repository = SourcesRepository(database)
        return self.sources_repository

    async def get_publish_service(self) -> PublishService:
        if self.publish_service is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.publish_service is None:
                    self.publish_service = PublishService(
                        repository=PublishRepository(database),
                        read_model_repository=ReadModelRepository(database),
                    )
        return self.publish_service

    async def get_assertion_repository(self) -> AssertionRepository:
        if self.assertion_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.assertion_repository is None:
                    self.assertion_repository = AssertionRepository(database)
        return self.assertion_repository

    async def get_claim_repository(self) -> ClaimRepository:
        if self.claim_repository is None:
            database = await self.get_database()
            async with self._init_lock:
                if self.claim_repository is None:
                    self.claim_repository = ClaimRepository(database)
        return self.claim_repository

    async def close(self) -> None:
        async def _close_async_resource(
            resource_name: str,
            resource: object | None,
            closer_name: str,
        ) -> None:
            if resource is None:
                return
            try:
                closer = getattr(resource, closer_name)
                await closer()
            except Exception:
                logger.warning(
                    "Failed to close app service resource %s",
                    resource_name,
                    exc_info=True,
                )

        broadcaster = self.alert_broadcaster
        self.alert_broadcaster = None
        await _close_async_resource("alert_broadcaster", broadcaster, "stop")

        embedding_service = self.embedding_service
        self.embedding_service = None
        await _close_async_resource("embedding_service", embedding_service, "close")

        sentiment_service = self.sentiment_service
        self.sentiment_service = None
        await _close_async_resource("sentiment_service", sentiment_service, "close")

        self.vector_store_manager = None
        self.theme_repository = None
        self.document_repository = None
        self.sentiment_aggregator = None
        self.ranking_service = None
        self.alert_repository = None
        self.narrative_repository = None
        self.causal_graph = None
        self.propagation_service = None
        self.feedback_repository = None
        self.graph_repository = None
        self.ner_service = None
        self.keywords_service = None
        self.pattern_extractor = None
        self.security_master_repository = None
        self.sources_repository = None
        self.publish_service = None
        self.assertion_repository = None
        self.claim_repository = None

        redis_client = self.redis_client
        self.redis_client = None
        await _close_async_resource("redis_client", redis_client, "close")

        database = self.database
        self.database = None
        await _close_async_resource("database", database, "close")


def ensure_app_services(app: FastAPI) -> AppServices:
    """Attach and return the app-scoped service container."""
    services = getattr(app.state, "services", None)
    if services is None:
        services = AppServices()
        app.state.services = services
    return services


def _get_services(request: Request) -> AppServices:
    return ensure_app_services(request.app)


async def get_database(request: Request) -> Database:
    return await _get_services(request).get_database()


async def get_redis_client(request: Request) -> redis.Redis:
    return await _get_services(request).get_redis_client()


async def get_embedding_service(request: Request) -> EmbeddingService:
    return await _get_services(request).get_embedding_service()


async def get_sentiment_service(request: Request) -> SentimentService:
    return await _get_services(request).get_sentiment_service()


async def get_vector_store_manager(request: Request) -> VectorStoreManager:
    return await _get_services(request).get_vector_store_manager()


async def get_theme_repository(request: Request) -> ThemeRepository:
    return await _get_services(request).get_theme_repository()


async def get_document_repository(request: Request) -> DocumentRepository:
    return await _get_services(request).get_document_repository()


async def get_ranking_service(request: Request) -> ThemeRankingService:
    return await _get_services(request).get_ranking_service()


async def get_alert_repository(request: Request) -> AlertRepository:
    return await _get_services(request).get_alert_repository()


async def get_narrative_repository(request: Request) -> NarrativeRepository:
    return await _get_services(request).get_narrative_repository()


async def get_feedback_repository(request: Request) -> FeedbackRepository:
    return await _get_services(request).get_feedback_repository()


def get_sentiment_aggregator(request: Request) -> SentimentAggregator:
    return _get_services(request).get_sentiment_aggregator()


async def get_graph_repository(request: Request) -> GraphRepository:
    return await _get_services(request).get_graph_repository()


async def get_causal_graph(request: Request) -> CausalGraph:
    return await _get_services(request).get_causal_graph()


async def get_propagation_service(request: Request) -> SentimentPropagation:
    return await _get_services(request).get_propagation_service()


async def get_alert_broadcaster(request: Request) -> AlertBroadcaster:
    return await _get_services(request).get_alert_broadcaster()


def get_ner_service(request: Request) -> NERService:
    return _get_services(request).get_ner_service()


def get_keywords_service(request: Request) -> KeywordsService:
    return _get_services(request).get_keywords_service()


def get_pattern_extractor(request: Request) -> PatternExtractor:
    return _get_services(request).get_pattern_extractor()


async def get_security_master_repository(request: Request) -> SecurityMasterRepository:
    return await _get_services(request).get_security_master_repository()


async def get_sources_repository(request: Request) -> SourcesRepository:
    return await _get_services(request).get_sources_repository()


async def get_publish_service(request: Request) -> PublishService:
    return await _get_services(request).get_publish_service()


async def get_assertion_repository(request: Request) -> AssertionRepository:
    return await _get_services(request).get_assertion_repository()


async def get_claim_repository(request: Request) -> ClaimRepository:
    return await _get_services(request).get_claim_repository()
