"""Tests for document explorer API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform


def _make_doc(
    doc_id: str = "doc_001",
    platform: Platform = Platform.TWITTER,
    content: str = "NVIDIA announces new GPU architecture for AI workloads",
    **kwargs,
) -> NormalizedDocument:
    """Helper to create a NormalizedDocument with sensible defaults."""
    return NormalizedDocument(
        id=doc_id,
        platform=platform,
        url=kwargs.pop("url", "https://example.com/article"),
        timestamp=kwargs.pop("timestamp", datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc)),
        fetched_at=kwargs.pop("fetched_at", datetime(2026, 2, 5, 10, 5, 0, tzinfo=timezone.utc)),
        author_id=kwargs.pop("author_id", "user_123"),
        author_name=kwargs.pop("author_name", "ChipAnalyst"),
        author_followers=kwargs.pop("author_followers", 5000),
        author_verified=kwargs.pop("author_verified", True),
        content=content,
        content_type=kwargs.pop("content_type", "post"),
        title=kwargs.pop("title", "NVIDIA GPU Launch"),
        engagement=kwargs.pop("engagement", EngagementMetrics(likes=100, shares=20, comments=5)),
        tickers_mentioned=kwargs.pop("tickers_mentioned", ["NVDA"]),
        entities_mentioned=kwargs.pop(
            "entities_mentioned",
            [{"type": "COMPANY", "normalized": "NVIDIA"}],
        ),
        keywords_extracted=kwargs.pop(
            "keywords_extracted",
            [{"text": "GPU", "score": 0.9}],
        ),
        events_extracted=kwargs.pop(
            "events_extracted",
            [{"event_type": "product_launch", "actor": "NVIDIA", "action": "announces", "object": "GPU", "time_ref": "Q2 2026"}],
        ),
        urls_mentioned=kwargs.pop("urls_mentioned", ["https://nvidia.com"]),
        spam_score=kwargs.pop("spam_score", 0.05),
        bot_probability=kwargs.pop("bot_probability", 0.01),
        authority_score=kwargs.pop("authority_score", 0.85),
        embedding=kwargs.pop("embedding", [0.1] * 768),
        embedding_minilm=kwargs.pop("embedding_minilm", None),
        sentiment=kwargs.pop("sentiment", {"label": "positive", "confidence": 0.92}),
        theme_ids=kwargs.pop("theme_ids", ["theme_abc123"]),
        **kwargs,
    )


def _make_list_record(**overrides):
    """Create a dict mimicking an asyncpg Record from list_documents."""
    rec = {
        "id": "doc_001",
        "platform": "twitter",
        "content_type": "post",
        "title": "NVIDIA GPU Launch",
        "content_preview": "NVIDIA announces new GPU architecture for AI workloads",
        "url": "https://example.com/article",
        "author_name": "ChipAnalyst",
        "author_verified": True,
        "author_followers": 5000,
        "tickers": ["NVDA"],
        "spam_score": 0.05,
        "authority_score": 0.85,
        "sentiment": {"label": "positive", "confidence": 0.92},
        "engagement": {"likes": 100, "shares": 20, "comments": 5},
        "theme_ids": ["theme_abc123"],
        "timestamp": datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
        "fetched_at": datetime(2026, 2, 5, 10, 5, 0, tzinfo=timezone.utc),
    }
    rec.update(overrides)
    return rec


# ── List Documents ───────────────────────────────────────────────────


class TestListDocuments:
    def test_empty_list(self, client, mock_doc_repo):
        resp = client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"] == []
        assert data["total"] == 0
        assert "latency_ms" in data

    def test_returns_items(self, client, mock_doc_repo):
        mock_doc_repo.list_documents.return_value = [_make_list_record()]
        mock_doc_repo.list_documents_count.return_value = 1

        resp = client.get("/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["documents"]) == 1
        assert data["total"] == 1

        doc = data["documents"][0]
        assert doc["document_id"] == "doc_001"
        assert doc["platform"] == "twitter"
        assert doc["tickers"] == ["NVDA"]
        assert doc["sentiment_label"] == "positive"
        assert doc["sentiment_confidence"] == 0.92
        assert doc["author_verified"] is True

    def test_pagination_params(self, client, mock_doc_repo):
        resp = client.get("/documents?limit=10&offset=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page_size"] == 10
        assert data["offset"] == 20

        # Verify repo was called with correct params
        mock_doc_repo.list_documents.assert_called_once()
        call_kwargs = mock_doc_repo.list_documents.call_args.kwargs
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 20

    def test_invalid_sort_returns_422(self, client, mock_doc_repo):
        resp = client.get("/documents?sort=malicious_field")
        assert resp.status_code == 422
        assert "Invalid sort field" in resp.json()["detail"]

    def test_invalid_order_returns_422(self, client, mock_doc_repo):
        resp = client.get("/documents?order=sideways")
        assert resp.status_code == 422
        assert "Invalid order" in resp.json()["detail"]

    def test_invalid_platform_returns_422(self, client, mock_doc_repo):
        resp = client.get("/documents?platform=facebook")
        assert resp.status_code == 422
        assert "Invalid platform" in resp.json()["detail"]

    def test_invalid_content_type_returns_422(self, client, mock_doc_repo):
        resp = client.get("/documents?content_type=video")
        assert resp.status_code == 422
        assert "Invalid content_type" in resp.json()["detail"]

    def test_valid_platform_filter(self, client, mock_doc_repo):
        resp = client.get("/documents?platform=twitter")
        assert resp.status_code == 200
        call_kwargs = mock_doc_repo.list_documents.call_args.kwargs
        assert call_kwargs["platform"] == "twitter"

    def test_valid_sort_order(self, client, mock_doc_repo):
        resp = client.get("/documents?sort=authority_score&order=asc")
        assert resp.status_code == 200
        call_kwargs = mock_doc_repo.list_documents.call_args.kwargs
        assert call_kwargs["sort"] == "authority_score"
        assert call_kwargs["order"] == "asc"

    def test_quality_filters(self, client, mock_doc_repo):
        resp = client.get("/documents?max_spam=0.3&min_authority=0.5")
        assert resp.status_code == 200
        call_kwargs = mock_doc_repo.list_documents.call_args.kwargs
        assert call_kwargs["max_spam"] == 0.3
        assert call_kwargs["min_authority"] == 0.5

    def test_text_search(self, client, mock_doc_repo):
        resp = client.get("/documents?q=nvidia+gpu")
        assert resp.status_code == 200
        call_kwargs = mock_doc_repo.list_documents.call_args.kwargs
        assert call_kwargs["q"] == "nvidia gpu"

    def test_null_sentiment_in_record(self, client, mock_doc_repo):
        """Records with no sentiment should have null label/confidence."""
        mock_doc_repo.list_documents.return_value = [
            _make_list_record(sentiment=None)
        ]
        mock_doc_repo.list_documents_count.return_value = 1

        resp = client.get("/documents")
        doc = resp.json()["documents"][0]
        assert doc["sentiment_label"] is None
        assert doc["sentiment_confidence"] is None


# ── Get Document Detail ──────────────────────────────────────────────


class TestGetDocument:
    def test_not_found(self, client, mock_doc_repo):
        resp = client.get("/documents/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_found(self, client, mock_doc_repo):
        mock_doc_repo.get_by_id.return_value = _make_doc()

        resp = client.get("/documents/doc_001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == "doc_001"
        assert data["platform"] == "twitter"
        assert data["content"] == "NVIDIA announces new GPU architecture for AI workloads"
        assert data["author_id"] == "user_123"
        assert data["bot_probability"] == 0.01
        assert "latency_ms" in data

    def test_embedding_flags(self, client, mock_doc_repo):
        """has_embedding/has_embedding_minilm reflect actual presence."""
        doc = _make_doc(embedding=[0.1] * 768, embedding_minilm=None)
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        data = resp.json()
        assert data["has_embedding"] is True
        assert data["has_embedding_minilm"] is False

    def test_embedding_both_present(self, client, mock_doc_repo):
        doc = _make_doc(embedding=[0.1] * 768, embedding_minilm=[0.2] * 384)
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        data = resp.json()
        assert data["has_embedding"] is True
        assert data["has_embedding_minilm"] is True

    def test_entity_normalization(self, client, mock_doc_repo):
        """Entities are remapped from {type, normalized} to {type, name}."""
        doc = _make_doc(
            entities_mentioned=[
                {"type": "COMPANY", "normalized": "NVIDIA"},
                {"type": "PRODUCT", "normalized": "H100"},
            ]
        )
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        entities = resp.json()["entities"]
        assert entities[0] == {"type": "COMPANY", "name": "NVIDIA"}
        assert entities[1] == {"type": "PRODUCT", "name": "H100"}

    def test_keyword_normalization(self, client, mock_doc_repo):
        """Keywords are remapped from {text, score} to {word, score}."""
        doc = _make_doc(keywords_extracted=[{"text": "GPU", "score": 0.9}])
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        keywords = resp.json()["keywords"]
        assert keywords[0] == {"word": "GPU", "score": 0.9}

    def test_event_normalization(self, client, mock_doc_repo):
        """Events are remapped to frontend shape."""
        doc = _make_doc(
            events_extracted=[
                {
                    "event_type": "capacity_expansion",
                    "actor": "TSMC",
                    "action": "builds",
                    "object": "fab",
                    "time_ref": "2027",
                }
            ]
        )
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        events = resp.json()["events"]
        assert events[0]["type"] == "capacity_expansion"
        assert events[0]["actor"] == "TSMC"

    def test_sentiment_fields(self, client, mock_doc_repo):
        doc = _make_doc(sentiment={"label": "negative", "confidence": 0.78})
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        data = resp.json()
        assert data["sentiment_label"] == "negative"
        assert data["sentiment_confidence"] == 0.78
        assert data["sentiment"] == {"label": "negative", "confidence": 0.78}

    def test_no_sentiment(self, client, mock_doc_repo):
        doc = _make_doc(sentiment=None)
        mock_doc_repo.get_by_id.return_value = doc

        resp = client.get("/documents/doc_001")
        data = resp.json()
        assert data["sentiment_label"] is None
        assert data["sentiment"] is None


# ── Document Stats ───────────────────────────────────────────────────


class TestGetDocumentStats:
    def test_returns_stats(self, client, mock_doc_repo):
        mock_doc_repo.get_document_stats.return_value = {
            "total_count": 1500,
            "platform_counts": [
                {"platform": "twitter", "count": 800},
                {"platform": "news", "count": 700},
            ],
            "embedding_coverage": {"finbert_pct": 0.85, "minilm_pct": 0.42},
            "sentiment_coverage": 0.73,
            "earliest_document": "2025-06-01T00:00:00+00:00",
            "latest_document": "2026-02-10T12:00:00+00:00",
        }

        resp = client.get("/documents/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 1500
        assert len(data["platform_counts"]) == 2
        assert data["platform_counts"][0]["platform"] == "twitter"
        assert data["embedding_coverage"]["finbert_pct"] == 0.85
        assert data["sentiment_coverage"] == 0.73
        assert data["earliest_document"] == "2025-06-01T00:00:00+00:00"
        assert "latency_ms" in data

    def test_empty_stats(self, client, mock_doc_repo):
        """Default mock returns zero stats successfully."""
        resp = client.get("/documents/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 0
        assert data["platform_counts"] == []

    def test_stats_not_captured_as_document_id(self, client, mock_doc_repo):
        """Verify /documents/stats is not matched by /documents/{document_id}."""
        # If route ordering is wrong, this would try to look up doc "stats"
        # and return 404 instead of proper stats response
        resp = client.get("/documents/stats")
        assert resp.status_code == 200
        assert "total_count" in resp.json()
        # get_by_id should NOT have been called
        mock_doc_repo.get_by_id.assert_not_called()
