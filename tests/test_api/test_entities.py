"""Tests for entity REST API endpoints."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import verify_api_key
from src.api.dependencies import get_document_repository, get_graph_repository


@pytest.fixture
def mock_doc_repo():
    """Mock DocumentRepository with entity methods."""
    repo = AsyncMock()
    repo.get_entity_counts = AsyncMock(return_value=[])
    repo.list_entities = AsyncMock(return_value=([], 0))
    repo.get_entity_detail = AsyncMock(return_value=None)
    repo.get_entity_sentiment = AsyncMock(return_value=None)
    repo.get_trending_entities = AsyncMock(return_value=[])
    repo.get_cooccurring_entities = AsyncMock(return_value=[])
    repo.get_documents_by_entity = AsyncMock(return_value=[])
    repo.merge_entity = AsyncMock(return_value=0)
    # _db for direct queries in stats endpoint
    repo._db = AsyncMock()
    repo._db.fetchval = AsyncMock(return_value=0)
    return repo


@pytest.fixture
def mock_graph_repo():
    """Mock GraphRepository."""
    repo = AsyncMock()
    repo.get_node = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def client(mock_doc_repo, mock_graph_repo):
    """FastAPI TestClient with dependency overrides for entities."""
    app = create_app()

    app.dependency_overrides[verify_api_key] = lambda: "test-key"
    app.dependency_overrides[get_document_repository] = lambda: mock_doc_repo
    app.dependency_overrides[get_graph_repository] = lambda: mock_graph_repo

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Feature gate ─────────────────────────────────


class TestFeatureGate:
    """Entity endpoints return 404 when ner_enabled=false."""

    @patch("src.api.routes.entities._get_settings")
    def test_entities_disabled(self, mock_settings, client):
        settings = MagicMock()
        settings.ner_enabled = False
        mock_settings.return_value = settings

        resp = client.get("/entities")
        assert resp.status_code == 404
        assert "ner_enabled" in resp.json()["detail"]

    @patch("src.api.routes.entities._get_settings")
    def test_stats_disabled(self, mock_settings, client):
        settings = MagicMock()
        settings.ner_enabled = False
        mock_settings.return_value = settings

        resp = client.get("/entities/stats")
        assert resp.status_code == 404


# ── GET /entities ────────────────────────────────


class TestListEntities:
    """Tests for the list entities endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_empty_list(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.list_entities.return_value = ([], 0)

        resp = client.get("/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities"] == []
        assert data["total"] == 0
        assert data["has_more"] is False
        assert "latency_ms" in data

    @patch("src.api.routes.entities._get_settings")
    def test_returns_entities(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.list_entities.return_value = (
            [
                {
                    "type": "COMPANY",
                    "normalized": "NVIDIA",
                    "mention_count": 42,
                    "first_seen": datetime(2026, 1, 1, tzinfo=timezone.utc),
                    "last_seen": datetime(2026, 2, 5, tzinfo=timezone.utc),
                },
                {
                    "type": "TICKER",
                    "normalized": "NVDA",
                    "mention_count": 30,
                    "first_seen": datetime(2026, 1, 5, tzinfo=timezone.utc),
                    "last_seen": datetime(2026, 2, 4, tzinfo=timezone.utc),
                },
            ],
            2,
        )

        resp = client.get("/entities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["entities"][0]["normalized"] == "NVIDIA"
        assert data["entities"][1]["type"] == "TICKER"

    @patch("src.api.routes.entities._get_settings")
    def test_entity_type_filter(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.list_entities.return_value = ([], 0)

        resp = client.get("/entities?entity_type=COMPANY")
        assert resp.status_code == 200
        mock_doc_repo.list_entities.assert_called_once_with(
            entity_type="COMPANY",
            search=None,
            sort="count",
            limit=50,
            offset=0,
        )

    @patch("src.api.routes.entities._get_settings")
    def test_invalid_entity_type(self, mock_settings, client):
        mock_settings.return_value = MagicMock(ner_enabled=True)

        resp = client.get("/entities?entity_type=BOGUS")
        assert resp.status_code == 422
        assert "Invalid entity_type" in resp.json()["detail"]

    @patch("src.api.routes.entities._get_settings")
    def test_search_filter(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.list_entities.return_value = ([], 0)

        resp = client.get("/entities?search=nvidia&sort=recent")
        assert resp.status_code == 200
        mock_doc_repo.list_entities.assert_called_once_with(
            entity_type=None,
            search="nvidia",
            sort="recent",
            limit=50,
            offset=0,
        )

    @patch("src.api.routes.entities._get_settings")
    def test_pagination(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.list_entities.return_value = ([], 100)

        resp = client.get("/entities?limit=10&offset=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_more"] is True
        mock_doc_repo.list_entities.assert_called_once_with(
            entity_type=None,
            search=None,
            sort="count",
            limit=10,
            offset=20,
        )

    @patch("src.api.routes.entities._get_settings")
    def test_server_error(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.list_entities.side_effect = RuntimeError("DB down")

        resp = client.get("/entities")
        assert resp.status_code == 500


# ── GET /entities/stats ──────────────────────────


class TestEntityStats:
    """Tests for the entity stats endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_stats_empty(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_entity_counts.return_value = []
        mock_doc_repo._db.fetchval.return_value = 0

        resp = client.get("/entities/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entities"] == 0
        assert data["documents_with_entities"] == 0
        assert data["by_type"] == {}

    @patch("src.api.routes.entities._get_settings")
    def test_stats_populated(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_entity_counts.return_value = [
            ("COMPANY", "NVIDIA", 50),
            ("COMPANY", "AMD", 30),
            ("TICKER", "NVDA", 25),
        ]
        mock_doc_repo._db.fetchval.return_value = 120

        resp = client.get("/entities/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_entities"] == 3
        assert data["documents_with_entities"] == 120
        assert data["by_type"]["COMPANY"] == 80
        assert data["by_type"]["TICKER"] == 25


# ── GET /entities/trending ───────────────────────


class TestTrendingEntities:
    """Tests for the trending entities endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_empty_trending(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_trending_entities.return_value = []

        resp = client.get("/entities/trending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trending"] == []

    @patch("src.api.routes.entities._get_settings")
    def test_trending_results(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_trending_entities.return_value = [
            {
                "type": "COMPANY",
                "normalized": "NVIDIA",
                "recent_count": 50,
                "baseline_count": 10,
                "spike_ratio": 5.0,
            },
        ]

        resp = client.get("/entities/trending")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["trending"]) == 1
        assert data["trending"][0]["spike_ratio"] == 5.0

    @patch("src.api.routes.entities._get_settings")
    def test_trending_custom_params(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_trending_entities.return_value = []

        resp = client.get("/entities/trending?hours_recent=48&hours_baseline=336&limit=5")
        assert resp.status_code == 200
        mock_doc_repo.get_trending_entities.assert_called_once_with(
            hours_recent=48,
            hours_baseline=336,
            limit=5,
        )


# ── GET /entities/{type}/{normalized} ────────────


class TestEntityDetail:
    """Tests for the entity detail endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_not_found(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True, graph_enabled=False)
        mock_doc_repo.get_entity_detail.return_value = None

        resp = client.get("/entities/COMPANY/NVIDIA")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    @patch("src.api.routes.entities._get_settings")
    def test_detail_found(self, mock_settings, client, mock_doc_repo, mock_graph_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True, graph_enabled=False)
        mock_doc_repo.get_entity_detail.return_value = {
            "type": "COMPANY",
            "normalized": "NVIDIA",
            "mention_count": 42,
            "first_seen": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "last_seen": datetime(2026, 2, 5, tzinfo=timezone.utc),
            "platforms": {"twitter": 20, "newsfilter": 22},
        }

        resp = client.get("/entities/COMPANY/NVIDIA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["normalized"] == "NVIDIA"
        assert data["mention_count"] == 42
        assert data["platforms"]["twitter"] == 20
        assert data["graph_node_id"] is None

    @patch("src.api.routes.entities._get_settings")
    def test_detail_with_graph_node(self, mock_settings, client, mock_doc_repo, mock_graph_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True, graph_enabled=True)
        mock_doc_repo.get_entity_detail.return_value = {
            "type": "COMPANY",
            "normalized": "NVIDIA",
            "mention_count": 42,
            "first_seen": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "last_seen": datetime(2026, 2, 5, tzinfo=timezone.utc),
            "platforms": {"twitter": 20},
        }
        mock_graph_repo.get_node.return_value = {"node_id": "nvidia"}

        resp = client.get("/entities/COMPANY/NVIDIA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["graph_node_id"] == "nvidia"


# ── GET /entities/{type}/{normalized}/cooccurrence


class TestEntityCooccurrence:
    """Tests for the co-occurrence endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_empty_cooccurrence(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_cooccurring_entities.return_value = []

        resp = client.get("/entities/COMPANY/NVIDIA/cooccurrence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entities"] == []

    @patch("src.api.routes.entities._get_settings")
    def test_cooccurrence_results(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_cooccurring_entities.return_value = [
            {"type": "TICKER", "normalized": "NVDA", "cooccurrence_count": 30, "jaccard": 0.75},
            {"type": "COMPANY", "normalized": "AMD", "cooccurrence_count": 15, "jaccard": 0.45},
        ]

        resp = client.get("/entities/COMPANY/NVIDIA/cooccurrence")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entities"]) == 2
        assert data["entities"][0]["jaccard"] == 0.75

    @patch("src.api.routes.entities._get_settings")
    def test_cooccurrence_params(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_cooccurring_entities.return_value = []

        resp = client.get("/entities/COMPANY/NVIDIA/cooccurrence?limit=5&min_count=3")
        assert resp.status_code == 200
        mock_doc_repo.get_cooccurring_entities.assert_called_once_with(
            entity_type="COMPANY",
            normalized="NVIDIA",
            limit=5,
            min_count=3,
        )


# ── GET /entities/{type}/{normalized}/sentiment ──


class TestEntitySentiment:
    """Tests for the entity sentiment endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_no_sentiment_data(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_entity_sentiment.return_value = None

        resp = client.get("/entities/COMPANY/NVIDIA/sentiment")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avg_score"] is None
        assert data["trend"] == "stable"

    @patch("src.api.routes.entities._get_settings")
    def test_sentiment_data(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.get_entity_sentiment.return_value = {
            "avg_score": 0.352,
            "pos_count": 20,
            "neg_count": 5,
            "neu_count": 10,
            "trend": "improving",
        }

        resp = client.get("/entities/COMPANY/NVIDIA/sentiment")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avg_score"] == 0.352
        assert data["pos_count"] == 20
        assert data["trend"] == "improving"


# ── POST /entities/{type}/{normalized}/merge ─────


class TestMergeEntity:
    """Tests for the entity merge endpoint."""

    @patch("src.api.routes.entities._get_settings")
    def test_merge_success(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.merge_entity.return_value = 15

        resp = client.post(
            "/entities/COMPANY/Nvidia Corp/merge",
            json={"to_type": "COMPANY", "to_normalized": "NVIDIA"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["affected_documents"] == 15
        assert data["merged_from"] == "COMPANY:Nvidia Corp"
        assert data["merged_to"] == "COMPANY:NVIDIA"

    @patch("src.api.routes.entities._get_settings")
    def test_merge_self(self, mock_settings, client):
        mock_settings.return_value = MagicMock(ner_enabled=True)

        resp = client.post(
            "/entities/COMPANY/NVIDIA/merge",
            json={"to_type": "COMPANY", "to_normalized": "NVIDIA"},
        )
        assert resp.status_code == 422
        assert "Cannot merge entity into itself" in resp.json()["detail"]

    @patch("src.api.routes.entities._get_settings")
    def test_merge_server_error(self, mock_settings, client, mock_doc_repo):
        mock_settings.return_value = MagicMock(ner_enabled=True)
        mock_doc_repo.merge_entity.side_effect = RuntimeError("DB down")

        resp = client.post(
            "/entities/COMPANY/NVIDIA/merge",
            json={"to_type": "TICKER", "to_normalized": "NVDA"},
        )
        assert resp.status_code == 500
