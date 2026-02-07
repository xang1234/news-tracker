"""Tests for theme REST API endpoints."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ingestion.schemas import EngagementMetrics, NormalizedDocument, Platform
from src.sentiment.aggregation import AggregatedSentiment
from tests.test_api.conftest import _make_metrics, _make_theme


# ── GET /themes ─────────────────────────────────────────


class TestListThemes:
    """Tests for the list themes endpoint."""

    def test_empty_list(self, client, mock_theme_repo):
        mock_theme_repo.get_all.return_value = []
        resp = client.get("/themes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["themes"] == []
        assert data["total"] == 0
        assert "latency_ms" in data

    def test_returns_themes(self, client, mock_theme_repo):
        themes = [_make_theme("t1", "theme_one"), _make_theme("t2", "theme_two")]
        mock_theme_repo.get_all.return_value = themes

        resp = client.get("/themes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["themes"][0]["theme_id"] == "t1"
        assert data["themes"][1]["theme_id"] == "t2"

    def test_centroid_excluded_by_default(self, client, mock_theme_repo):
        mock_theme_repo.get_all.return_value = [_make_theme()]

        resp = client.get("/themes")
        data = resp.json()
        assert data["themes"][0]["centroid"] is None

    def test_centroid_included_when_requested(self, client, mock_theme_repo):
        theme = _make_theme(centroid=np.ones(768, dtype=np.float32))
        mock_theme_repo.get_all.return_value = [theme]

        resp = client.get("/themes?include_centroid=true")
        data = resp.json()
        centroid = data["themes"][0]["centroid"]
        assert centroid is not None
        assert len(centroid) == 768

    def test_lifecycle_stage_filter(self, client, mock_theme_repo):
        mock_theme_repo.get_all.return_value = []

        resp = client.get("/themes?lifecycle_stage=accelerating")
        assert resp.status_code == 200
        mock_theme_repo.get_all.assert_called_once_with(
            lifecycle_stages=["accelerating"], limit=50 + 0,
        )

    def test_pagination(self, client, mock_theme_repo):
        themes = [_make_theme(f"t{i}") for i in range(5)]
        mock_theme_repo.get_all.return_value = themes

        resp = client.get("/themes?limit=2&offset=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["themes"][0]["theme_id"] == "t1"
        assert data["themes"][1]["theme_id"] == "t2"

    def test_theme_fields(self, client, mock_theme_repo):
        theme = _make_theme()
        mock_theme_repo.get_all.return_value = [theme]

        resp = client.get("/themes")
        item = resp.json()["themes"][0]
        assert item["name"] == "gpu_nvidia_hbm"
        assert item["top_keywords"] == ["gpu", "nvidia", "hbm"]
        assert item["top_tickers"] == ["NVDA", "AMD"]
        assert item["lifecycle_stage"] == "emerging"
        assert item["document_count"] == 42
        assert item["description"] == "GPU and NVIDIA HBM memory theme"
        assert item["metadata"] == {"bertopic_topic_id": 3}
        assert "created_at" in item
        assert "updated_at" in item


# ── GET /themes/{theme_id} ──────────────────────────────


class TestGetTheme:
    """Tests for the get theme detail endpoint."""

    def test_found(self, client, mock_theme_repo):
        theme = _make_theme()
        mock_theme_repo.get_by_id.return_value = theme

        resp = client.get("/themes/theme_abc123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["theme"]["theme_id"] == "theme_abc123"
        assert "latency_ms" in data

    def test_not_found(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = None

        resp = client.get("/themes/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_centroid_opt_in(self, client, mock_theme_repo):
        theme = _make_theme(centroid=np.ones(768, dtype=np.float32))
        mock_theme_repo.get_by_id.return_value = theme

        # Without centroid
        resp = client.get("/themes/theme_abc123")
        assert resp.json()["theme"]["centroid"] is None

        # With centroid
        resp = client.get("/themes/theme_abc123?include_centroid=true")
        assert resp.json()["theme"]["centroid"] is not None
        assert len(resp.json()["theme"]["centroid"]) == 768


# ── GET /themes/{theme_id}/documents ─────────────────────


class TestGetThemeDocuments:
    """Tests for the get theme documents endpoint."""

    def _make_doc(self, doc_id: str = "news_article1") -> NormalizedDocument:
        """Create a minimal NormalizedDocument."""
        return NormalizedDocument(
            id=doc_id,
            platform=Platform.NEWS,
            url="https://example.com/article",
            timestamp=datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
            fetched_at=datetime(2026, 2, 5, 10, 0, 0, tzinfo=timezone.utc),
            author_id="author1",
            author_name="Jane Doe",
            content="NVIDIA announced new HBM3E support for H200 GPUs with improved memory bandwidth.",
            content_type="article",
            title="NVIDIA HBM3E Announcement",
            engagement=EngagementMetrics(likes=100, shares=50, comments=20),
            tickers_mentioned=["NVDA"],
            authority_score=0.85,
            sentiment={"label": "positive", "confidence": 0.92, "scores": {"positive": 0.92, "negative": 0.03, "neutral": 0.05}},
            theme_ids=["theme_abc123"],
        )

    def test_returns_documents(self, client, mock_theme_repo, mock_doc_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_documents_by_theme.return_value = [self._make_doc()]

        resp = client.get("/themes/theme_abc123/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["theme_id"] == "theme_abc123"
        doc = data["documents"][0]
        assert doc["document_id"] == "news_article1"
        assert doc["platform"] == "news"
        assert doc["title"] == "NVIDIA HBM3E Announcement"
        assert doc["tickers"] == ["NVDA"]
        assert doc["authority_score"] == 0.85
        assert doc["sentiment_label"] == "positive"
        assert doc["sentiment_confidence"] == 0.92

    def test_content_preview_truncated(self, client, mock_theme_repo, mock_doc_repo):
        doc = self._make_doc()
        doc.content = "X" * 500
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_documents_by_theme.return_value = [doc]

        resp = client.get("/themes/theme_abc123/documents")
        preview = resp.json()["documents"][0]["content_preview"]
        assert len(preview) == 300

    def test_theme_not_found(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = None

        resp = client.get("/themes/nonexistent/documents")
        assert resp.status_code == 404

    def test_platform_filter(self, client, mock_theme_repo, mock_doc_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_documents_by_theme.return_value = []

        resp = client.get("/themes/theme_abc123/documents?platform=twitter")
        assert resp.status_code == 200
        mock_doc_repo.get_documents_by_theme.assert_called_once_with(
            theme_id="theme_abc123",
            limit=50,
            offset=0,
            platform="twitter",
            min_authority=None,
        )

    def test_min_authority_filter(self, client, mock_theme_repo, mock_doc_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_documents_by_theme.return_value = []

        resp = client.get("/themes/theme_abc123/documents?min_authority=0.5")
        assert resp.status_code == 200
        mock_doc_repo.get_documents_by_theme.assert_called_once_with(
            theme_id="theme_abc123",
            limit=50,
            offset=0,
            platform=None,
            min_authority=0.5,
        )

    def test_pagination(self, client, mock_theme_repo, mock_doc_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_documents_by_theme.return_value = []

        resp = client.get("/themes/theme_abc123/documents?limit=10&offset=20")
        assert resp.status_code == 200
        mock_doc_repo.get_documents_by_theme.assert_called_once_with(
            theme_id="theme_abc123",
            limit=10,
            offset=20,
            platform=None,
            min_authority=None,
        )

    def test_doc_without_sentiment(self, client, mock_theme_repo, mock_doc_repo):
        doc = self._make_doc()
        doc.sentiment = None
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_documents_by_theme.return_value = [doc]

        resp = client.get("/themes/theme_abc123/documents")
        item = resp.json()["documents"][0]
        assert item["sentiment_label"] is None
        assert item["sentiment_confidence"] is None


# ── GET /themes/{theme_id}/sentiment ─────────────────────


class TestGetThemeSentiment:
    """Tests for the theme sentiment aggregation endpoint."""

    def test_aggregation_flow(self, client, mock_theme_repo, mock_doc_repo, mock_aggregator):
        mock_theme_repo.get_by_id.return_value = _make_theme()

        # Return some sentiment rows
        now = datetime.now(timezone.utc)
        mock_doc_repo.get_sentiments_for_theme.return_value = [
            {
                "document_id": "d1",
                "timestamp": now - timedelta(hours=2),
                "platform": "news",
                "authority_score": 0.8,
                "sentiment": {
                    "label": "positive",
                    "confidence": 0.9,
                    "scores": {"positive": 0.9, "negative": 0.05, "neutral": 0.05},
                },
            },
            {
                "document_id": "d2",
                "timestamp": now - timedelta(hours=5),
                "platform": "twitter",
                "authority_score": 0.6,
                "sentiment": {
                    "label": "negative",
                    "confidence": 0.7,
                    "scores": {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
                },
            },
        ]

        # Mock aggregator return value
        mock_aggregator.aggregate_theme_sentiment.return_value = AggregatedSentiment(
            theme_id="theme_abc123",
            ticker=None,
            window_start=now - timedelta(days=7),
            window_end=now,
            document_count=2,
            bullish_ratio=0.55,
            bearish_ratio=0.35,
            neutral_ratio=0.10,
            avg_confidence=0.80,
            avg_authority=0.70,
            sentiment_velocity=0.02,
            extreme_sentiment=None,
        )

        resp = client.get("/themes/theme_abc123/sentiment?window_days=7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["theme_id"] == "theme_abc123"
        assert data["bullish_ratio"] == 0.55
        assert data["bearish_ratio"] == 0.35
        assert data["neutral_ratio"] == 0.10
        assert data["document_count"] == 2
        assert data["sentiment_velocity"] == 0.02
        assert data["extreme_sentiment"] is None
        assert "latency_ms" in data

    def test_theme_not_found(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = None

        resp = client.get("/themes/nonexistent/sentiment")
        assert resp.status_code == 404

    def test_skips_invalid_sentiment_rows(self, client, mock_theme_repo, mock_doc_repo, mock_aggregator):
        mock_theme_repo.get_by_id.return_value = _make_theme()

        # Mix of valid and invalid rows
        now = datetime.now(timezone.utc)
        mock_doc_repo.get_sentiments_for_theme.return_value = [
            {
                "document_id": "d1",
                "timestamp": now,
                "platform": "news",
                "authority_score": 0.8,
                "sentiment": {"label": "positive", "confidence": 0.9, "scores": {}},
            },
            {
                "document_id": "d2",
                "timestamp": now,
                "platform": "news",
                "authority_score": 0.5,
                "sentiment": {"label": "invalid_label", "confidence": 0.5, "scores": {}},
            },
            {
                "document_id": "d3",
                "timestamp": now,
                "platform": "news",
                "authority_score": 0.5,
                "sentiment": None,  # Missing sentiment
            },
        ]

        mock_aggregator.aggregate_theme_sentiment.return_value = AggregatedSentiment(
            theme_id="theme_abc123",
            ticker=None,
            window_start=now - timedelta(days=7),
            window_end=now,
            document_count=1,
            bullish_ratio=1.0,
            bearish_ratio=0.0,
            neutral_ratio=0.0,
            avg_confidence=0.9,
            avg_authority=0.8,
        )

        resp = client.get("/themes/theme_abc123/sentiment")
        assert resp.status_code == 200

        # Verify only 1 valid DocumentSentiment was passed to aggregator
        call_args = mock_aggregator.aggregate_theme_sentiment.call_args
        doc_sentiments = call_args.kwargs.get("document_sentiments") or call_args[1].get("document_sentiments")
        if doc_sentiments is None:
            # positional arg
            doc_sentiments = call_args[0][2]
        assert len(doc_sentiments) == 1
        assert doc_sentiments[0].document_id == "d1"

    def test_empty_sentiment(self, client, mock_theme_repo, mock_doc_repo, mock_aggregator):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_doc_repo.get_sentiments_for_theme.return_value = []

        now = datetime.now(timezone.utc)
        mock_aggregator.aggregate_theme_sentiment.return_value = AggregatedSentiment(
            theme_id="theme_abc123",
            ticker=None,
            window_start=now - timedelta(days=7),
            window_end=now,
            document_count=0,
            bullish_ratio=0.0,
            bearish_ratio=0.0,
            neutral_ratio=0.0,
            avg_confidence=0.0,
            avg_authority=None,
        )

        resp = client.get("/themes/theme_abc123/sentiment")
        assert resp.status_code == 200
        assert resp.json()["document_count"] == 0


# ── GET /themes/{theme_id}/metrics ───────────────────────


class TestGetThemeMetrics:
    """Tests for the theme metrics endpoint."""

    def test_returns_metrics(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        metrics = [
            _make_metrics(target_date=date(2026, 2, 3)),
            _make_metrics(target_date=date(2026, 2, 4)),
            _make_metrics(target_date=date(2026, 2, 5)),
        ]
        mock_theme_repo.get_metrics_range.return_value = metrics

        resp = client.get("/themes/theme_abc123/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["theme_id"] == "theme_abc123"
        assert data["metrics"][0]["date"] == "2026-02-03"
        assert data["metrics"][0]["document_count"] == 10
        assert data["metrics"][0]["sentiment_score"] == 0.3
        assert data["metrics"][0]["bullish_ratio"] == 0.6

    def test_theme_not_found(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = None

        resp = client.get("/themes/nonexistent/metrics")
        assert resp.status_code == 404

    def test_custom_date_range(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_theme_repo.get_metrics_range.return_value = []

        resp = client.get("/themes/theme_abc123/metrics?start_date=2026-01-01&end_date=2026-01-31")
        assert resp.status_code == 200
        mock_theme_repo.get_metrics_range.assert_called_once_with(
            theme_id="theme_abc123",
            start=date(2026, 1, 1),
            end=date(2026, 1, 31),
        )

    def test_default_date_range(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_theme_repo.get_metrics_range.return_value = []

        resp = client.get("/themes/theme_abc123/metrics")
        assert resp.status_code == 200

        # Should default to last 30 days
        call_args = mock_theme_repo.get_metrics_range.call_args
        assert call_args.kwargs["end"] == date.today()
        assert call_args.kwargs["start"] == date.today() - timedelta(days=30)

    def test_empty_metrics(self, client, mock_theme_repo):
        mock_theme_repo.get_by_id.return_value = _make_theme()
        mock_theme_repo.get_metrics_range.return_value = []

        resp = client.get("/themes/theme_abc123/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["metrics"] == []


# ── GET /themes/ranked ─────────────────────────────────


class TestRankedThemes:
    """Tests for the ranked themes endpoint."""

    def test_empty_ranked(self, client, mock_ranking_service):
        mock_ranking_service.get_actionable.return_value = []

        resp = client.get("/themes/ranked")
        assert resp.status_code == 200
        data = resp.json()
        assert data["themes"] == []
        assert data["total"] == 0
        assert data["strategy"] == "swing"
        assert "latency_ms" in data

    def test_returns_ranked_themes(self, client, mock_ranking_service):
        from src.themes.ranking import RankedTheme

        theme = _make_theme("t1", "ranked_theme", lifecycle_stage="accelerating")
        ranked = RankedTheme(
            theme_id="t1",
            theme=theme,
            score=8.5,
            tier=1,
            components={
                "volume_component": 2.0,
                "compellingness_component": 2.5,
                "lifecycle_multiplier": 1.2,
                "volume_zscore": 3.0,
                "strategy": "swing",
            },
        )
        mock_ranking_service.get_actionable.return_value = [ranked]

        resp = client.get("/themes/ranked")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        item = data["themes"][0]
        assert item["theme"]["theme_id"] == "t1"
        assert item["score"] == 8.5
        assert item["tier"] == 1
        assert "volume_component" in item["components"]

    def test_strategy_param(self, client, mock_ranking_service):
        mock_ranking_service.get_actionable.return_value = []

        resp = client.get("/themes/ranked?strategy=position")
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "position"
        mock_ranking_service.get_actionable.assert_called_once_with(
            strategy="position", max_tier=3,
        )

    def test_max_tier_param(self, client, mock_ranking_service):
        mock_ranking_service.get_actionable.return_value = []

        resp = client.get("/themes/ranked?max_tier=1")
        assert resp.status_code == 200
        mock_ranking_service.get_actionable.assert_called_once_with(
            strategy="swing", max_tier=1,
        )

    def test_limit_param(self, client, mock_ranking_service):
        from src.themes.ranking import RankedTheme

        # Return 5 ranked themes
        ranked = [
            RankedTheme(
                theme_id=f"t{i}",
                theme=_make_theme(f"t{i}"),
                score=float(10 - i),
                tier=2,
                components={},
            )
            for i in range(5)
        ]
        mock_ranking_service.get_actionable.return_value = ranked

        resp = client.get("/themes/ranked?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_invalid_strategy(self, client, mock_ranking_service):
        resp = client.get("/themes/ranked?strategy=invalid")
        assert resp.status_code == 400
        assert "Invalid strategy" in resp.json()["detail"]

    def test_ranked_does_not_conflict_with_theme_id(self, client, mock_theme_repo):
        """Ensure /themes/ranked is not captured by /themes/{theme_id}."""
        # If routing is wrong, this would try to look up theme_id="ranked"
        # and return 404. Instead it should return 200 from the ranked endpoint.
        resp = client.get("/themes/ranked")
        assert resp.status_code == 200
