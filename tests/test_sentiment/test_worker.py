"""Tests for SentimentWorker."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.sentiment.config import SentimentConfig
from src.sentiment.queue import SentimentJob, SentimentQueue
from src.sentiment.service import SentimentService
from src.sentiment.worker import SentimentWorker
from src.storage.database import Database
from src.storage.repository import DocumentRepository


def _make_document(
    doc_id: str,
    *,
    title: str | None = None,
    content: str = "Body text",
    sentiment: dict | None = None,
    entities_mentioned: list[dict] | None = None,
) -> NormalizedDocument:
    """Create a document for worker tests."""
    return NormalizedDocument(
        id=doc_id,
        platform=Platform.NEWS,
        url=f"https://example.com/{doc_id}",
        timestamp=datetime.now(UTC),
        author_id=f"author_{doc_id}",
        author_name=f"Author {doc_id}",
        content=content,
        content_type="article",
        title=title,
        engagement=EngagementMetrics(),
        sentiment=sentiment,
        entities_mentioned=entities_mentioned or [],
    )


@pytest.fixture
def mock_queue() -> AsyncMock:
    """Create a mock sentiment queue."""
    queue = AsyncMock(spec=SentimentQueue)
    queue.ack = AsyncMock()
    queue.nack = AsyncMock()
    return queue


@pytest.fixture
def mock_database() -> AsyncMock:
    """Create a mock database."""
    return AsyncMock(spec=Database)


@pytest.fixture
def mock_sentiment_service() -> AsyncMock:
    """Create a mock sentiment service."""
    service = AsyncMock(spec=SentimentService)
    service.analyze = AsyncMock()
    service.analyze_batch = AsyncMock()
    service.analyze_with_entities = AsyncMock()
    return service


class TestSentimentWorker:
    """Tests for sentiment worker batch behavior."""

    @pytest.mark.asyncio
    async def test_fetch_documents_uses_bulk_repository_call(
        self,
        mock_database,
    ):
        """Should fetch documents with one get_by_ids call."""
        worker = SentimentWorker(database=mock_database)

        expected_documents = [
            _make_document("doc_plain", title="Plain title", content="Plain content"),
            _make_document("doc_entity", title="Entity title", content="Entity content"),
        ]
        mock_repo = AsyncMock(spec=DocumentRepository)
        mock_repo.get_by_ids = AsyncMock(return_value=expected_documents)
        mock_repo.get_by_id = AsyncMock()
        worker._repository = mock_repo

        documents = await worker._fetch_documents(["doc_plain", "doc_entity"])

        assert documents == expected_documents
        mock_repo.get_by_ids.assert_awaited_once_with(["doc_plain", "doc_entity"])
        mock_repo.get_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_batch_uses_batch_and_entity_lanes(
        self,
        mock_queue,
        mock_database,
        mock_sentiment_service,
    ):
        """Should batch plain docs and keep entity sentiment per document."""
        config = SentimentConfig(enable_entity_sentiment=True, batch_size=4)
        worker = SentimentWorker(
            queue=mock_queue,
            database=mock_database,
            sentiment_service=mock_sentiment_service,
            config=config,
        )

        plain_doc = _make_document(
            "doc_plain",
            title="Plain title",
            content="Plain content",
        )
        entity_doc = _make_document(
            "doc_entity",
            title="Entity title",
            content="Entity content",
            entities_mentioned=[{"text": "NVIDIA", "type": "COMPANY"}],
        )

        plain_result = {"label": "positive", "confidence": 0.91}
        entity_result = {
            "label": "neutral",
            "confidence": 0.72,
            "entity_sentiments": [{"entity": "NVIDIA", "label": "positive"}],
        }
        mock_sentiment_service.analyze_batch.return_value = [plain_result]
        mock_sentiment_service.analyze_with_entities.return_value = entity_result

        mock_repo = AsyncMock(spec=DocumentRepository)
        mock_repo.get_by_ids = AsyncMock(return_value=[plain_doc, entity_doc])
        mock_repo.update_sentiment = AsyncMock(return_value=True)
        worker._repository = mock_repo

        metrics = MagicMock()
        jobs = [
            SentimentJob(message_id="msg_plain", document_id="doc_plain"),
            SentimentJob(message_id="msg_entity", document_id="doc_entity"),
        ]

        with patch("src.sentiment.worker.get_metrics", return_value=metrics):
            await worker._process_batch(jobs)

        mock_sentiment_service.analyze_batch.assert_awaited_once_with(
            ["Plain title. Plain content"]
        )
        mock_sentiment_service.analyze.assert_not_called()
        mock_sentiment_service.analyze_with_entities.assert_awaited_once_with(
            "Entity title. Entity content",
            entity_doc.entities_mentioned,
        )
        assert mock_repo.update_sentiment.await_args_list == [
            call("doc_plain", plain_result),
            call("doc_entity", entity_result),
        ]
        assert mock_queue.ack.await_args_list == [call("msg_plain"), call("msg_entity")]
        mock_queue.nack.assert_not_called()
        assert mock_repo.update_sentiment.await_count == 2
        assert mock_queue.ack.await_count == 2
