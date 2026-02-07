"""Tests for ModelVersion, ModelVersionRepository, and version ID generation."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.backtest.model_versions import (
    ModelVersion,
    ModelVersionRepository,
    compute_version_id,
    create_version_from_settings,
)


class TestComputeVersionId:
    """Test deterministic version ID generation."""

    def test_deterministic(self) -> None:
        """Same config always produces the same ID."""
        config = {"embedding_model": "ProsusAI/finbert", "batch_size": 32}
        id1 = compute_version_id(config)
        id2 = compute_version_id(config)
        assert id1 == id2

    def test_prefix(self) -> None:
        """Version IDs start with 'mv_'."""
        config = {"model": "test"}
        version_id = compute_version_id(config)
        assert version_id.startswith("mv_")

    def test_length(self) -> None:
        """Version IDs are mv_ + 12 hex chars = 15 chars total."""
        config = {"model": "test"}
        version_id = compute_version_id(config)
        assert len(version_id) == 15

    def test_different_configs_different_ids(self) -> None:
        """Different configs produce different IDs."""
        id1 = compute_version_id({"model": "finbert"})
        id2 = compute_version_id({"model": "minilm"})
        assert id1 != id2

    def test_key_order_irrelevant(self) -> None:
        """Dict key order doesn't affect the hash (sort_keys=True)."""
        id1 = compute_version_id({"a": 1, "b": 2})
        id2 = compute_version_id({"b": 2, "a": 1})
        assert id1 == id2


class TestCreateVersionFromSettings:
    """Test the factory function for creating ModelVersion from settings."""

    def test_basic_creation(self) -> None:
        version = create_version_from_settings(
            embedding_model="ProsusAI/finbert",
            clustering_config={"min_cluster_size": 10},
        )
        assert version.version_id.startswith("mv_")
        assert version.embedding_model == "ProsusAI/finbert"
        assert version.clustering_config == {"min_cluster_size": 10}

    def test_with_full_settings(self) -> None:
        settings = {"embedding_model_name": "ProsusAI/finbert", "debug": False}
        version = create_version_from_settings(
            embedding_model="ProsusAI/finbert",
            clustering_config={},
            full_settings=settings,
        )
        assert version.config_snapshot == settings

    def test_with_description(self) -> None:
        version = create_version_from_settings(
            embedding_model="ProsusAI/finbert",
            clustering_config={},
            description="v1 baseline",
        )
        assert version.description == "v1 baseline"

    def test_deterministic_id(self) -> None:
        """Same inputs produce the same version_id."""
        v1 = create_version_from_settings(
            embedding_model="ProsusAI/finbert",
            clustering_config={"min_cluster_size": 10},
        )
        v2 = create_version_from_settings(
            embedding_model="ProsusAI/finbert",
            clustering_config={"min_cluster_size": 10},
        )
        assert v1.version_id == v2.version_id


class TestModelVersionRepository:
    """Test ModelVersionRepository CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_upsert(
        self,
        mock_database: AsyncMock,
        sample_model_version: ModelVersion,
        sample_model_version_row: dict,
    ) -> None:
        """create() uses INSERT ... ON CONFLICT DO UPDATE."""
        mock_database.fetchrow.return_value = sample_model_version_row
        repo = ModelVersionRepository(mock_database)

        result = await repo.create(sample_model_version)

        sql = mock_database.fetchrow.call_args[0][0]
        assert "INSERT INTO model_versions" in sql
        assert "ON CONFLICT (version_id) DO UPDATE" in sql
        assert result.version_id == "mv_abc123def456"

    @pytest.mark.asyncio
    async def test_create_params(
        self,
        mock_database: AsyncMock,
        sample_model_version: ModelVersion,
        sample_model_version_row: dict,
    ) -> None:
        """create() passes correct params including JSON serialization."""
        mock_database.fetchrow.return_value = sample_model_version_row
        repo = ModelVersionRepository(mock_database)

        await repo.create(sample_model_version)

        args = mock_database.fetchrow.call_args[0]
        assert args[1] == sample_model_version.version_id
        assert args[2] == sample_model_version.embedding_model
        # clustering_config and config_snapshot are JSON strings
        assert json.loads(args[3]) == sample_model_version.clustering_config
        assert json.loads(args[4]) == sample_model_version.config_snapshot

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self,
        mock_database: AsyncMock,
        sample_model_version_row: dict,
    ) -> None:
        mock_database.fetchrow.return_value = sample_model_version_row
        repo = ModelVersionRepository(mock_database)

        result = await repo.get_by_id("mv_abc123def456")

        assert result is not None
        assert result.version_id == "mv_abc123def456"
        assert result.embedding_model == "ProsusAI/finbert"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self,
        mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchrow.return_value = None
        repo = ModelVersionRepository(mock_database)

        result = await repo.get_by_id("mv_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_latest(
        self,
        mock_database: AsyncMock,
        sample_model_version_row: dict,
    ) -> None:
        mock_database.fetchrow.return_value = sample_model_version_row
        repo = ModelVersionRepository(mock_database)

        result = await repo.get_latest()

        sql = mock_database.fetchrow.call_args[0][0]
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT 1" in sql
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_latest_none(
        self,
        mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchrow.return_value = None
        repo = ModelVersionRepository(mock_database)

        result = await repo.get_latest()
        assert result is None

    @pytest.mark.asyncio
    async def test_list_versions(
        self,
        mock_database: AsyncMock,
        sample_model_version_row: dict,
    ) -> None:
        mock_database.fetch.return_value = [sample_model_version_row]
        repo = ModelVersionRepository(mock_database)

        results = await repo.list_versions(limit=10)

        sql = mock_database.fetch.call_args[0][0]
        assert "ORDER BY created_at DESC" in sql
        assert "LIMIT $1" in sql
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_jsonb_parsing(
        self,
        mock_database: AsyncMock,
        sample_model_version_row: dict,
    ) -> None:
        """JSONB string fields are parsed to Python dicts."""
        mock_database.fetchrow.return_value = sample_model_version_row
        repo = ModelVersionRepository(mock_database)

        result = await repo.get_by_id("mv_abc123def456")

        assert isinstance(result.clustering_config, dict)
        assert isinstance(result.config_snapshot, dict)
