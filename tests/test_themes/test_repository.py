"""Tests for ThemeRepository CRUD operations."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, call

import numpy as np
import pytest

from src.themes.repository import (
    ThemeRepository,
    _centroid_to_pgvector,
    _parse_centroid,
    _row_to_theme,
)
from src.themes.schemas import Theme


class TestCreate:
    """Test ThemeRepository.create()."""

    @pytest.mark.asyncio
    async def test_insert_sql_and_params(
        self,
        mock_database: AsyncMock,
        sample_theme: Theme,
        sample_db_row: dict,
    ) -> None:
        """create() passes correct SQL and serialised params."""
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)

        result = await repo.create(sample_theme)

        # Verify fetchrow was called with INSERT ... RETURNING *
        args = mock_database.fetchrow.call_args
        sql = args[0][0]
        assert "INSERT INTO themes" in sql
        assert "RETURNING *" in sql

        # Verify positional params
        params = args[0][1:]
        assert params[0] == sample_theme.theme_id
        assert params[1] == sample_theme.name
        assert params[2] == sample_theme.description

        # centroid should be pgvector string
        assert params[3].startswith("[")
        assert params[3].endswith("]")

        # TEXT[] passed as plain lists
        assert params[4] == sample_theme.top_keywords
        assert params[5] == sample_theme.top_tickers

        # JSONB passed as json strings
        assert json.loads(params[6]) == sample_theme.top_entities
        assert json.loads(params[9]) == sample_theme.metadata

        assert params[7] == sample_theme.document_count
        assert params[8] == sample_theme.lifecycle_stage

        assert result.theme_id == sample_theme.theme_id

    @pytest.mark.asyncio
    async def test_centroid_formatting(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        """Centroid is converted to pgvector string format."""
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)
        theme = Theme(
            theme_id="theme_test",
            name="test",
            centroid=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
        await repo.create(theme)

        params = mock_database.fetchrow.call_args[0]
        centroid_param = params[4]  # $4 = centroid
        assert centroid_param == _centroid_to_pgvector(theme.centroid)


class TestGetById:
    """Test ThemeRepository.get_by_id()."""

    @pytest.mark.asyncio
    async def test_found(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)

        result = await repo.get_by_id("theme_a1b2c3d4e5f6")

        assert result is not None
        assert result.theme_id == "theme_a1b2c3d4e5f6"
        assert result.name == "gpu_nvidia_architecture"

    @pytest.mark.asyncio
    async def test_not_found(self, mock_database: AsyncMock) -> None:
        mock_database.fetchrow.return_value = None
        repo = ThemeRepository(mock_database)

        result = await repo.get_by_id("theme_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_jsonb_parsing(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        """JSONB string fields are parsed to Python objects."""
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)

        result = await repo.get_by_id("theme_a1b2c3d4e5f6")
        assert isinstance(result.top_entities, list)
        assert result.top_entities[0]["type"] == "COMPANY"
        assert isinstance(result.metadata, dict)
        assert result.metadata["bertopic_topic_id"] == 3


class TestGetAll:
    """Test ThemeRepository.get_all()."""

    @pytest.mark.asyncio
    async def test_no_filter(self, mock_database: AsyncMock) -> None:
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)

        result = await repo.get_all()

        sql = mock_database.fetch.call_args[0][0]
        assert "WHERE" not in sql
        assert "ORDER BY updated_at DESC" in sql
        assert "LIMIT" in sql
        assert result == []

    @pytest.mark.asyncio
    async def test_with_lifecycle_stages(
        self, mock_database: AsyncMock,
    ) -> None:
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)

        await repo.get_all(lifecycle_stages=["emerging", "accelerating"])

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "lifecycle_stage = ANY($1)" in sql
        assert args[1] == ["emerging", "accelerating"]

    @pytest.mark.asyncio
    async def test_respects_limit(self, mock_database: AsyncMock) -> None:
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)

        await repo.get_all(limit=5)

        args = mock_database.fetch.call_args[0]
        assert args[1] == 5  # limit param


class TestUpdate:
    """Test ThemeRepository.update()."""

    @pytest.mark.asyncio
    async def test_single_field(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)

        result = await repo.update(
            "theme_a1b2c3d4e5f6", {"name": "new_name"},
        )

        sql = mock_database.fetchrow.call_args[0][0]
        assert "SET name = $1" in sql
        assert "WHERE theme_id = $2" in sql
        assert result.theme_id == "theme_a1b2c3d4e5f6"

    @pytest.mark.asyncio
    async def test_multiple_fields(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)

        await repo.update(
            "theme_a1b2c3d4e5f6",
            {"name": "updated", "document_count": 50},
        )

        args = mock_database.fetchrow.call_args[0]
        sql = args[0]
        assert "name = $1" in sql
        assert "document_count = $2" in sql
        assert "WHERE theme_id = $3" in sql
        assert args[1] == "updated"
        assert args[2] == 50

    @pytest.mark.asyncio
    async def test_invalid_field_raises(
        self, mock_database: AsyncMock,
    ) -> None:
        repo = ThemeRepository(mock_database)

        with pytest.raises(ValueError, match="Invalid fields"):
            await repo.update("theme_x", {"centroid": np.zeros(768)})

    @pytest.mark.asyncio
    async def test_empty_updates_raises(
        self, mock_database: AsyncMock,
    ) -> None:
        repo = ThemeRepository(mock_database)

        with pytest.raises(ValueError, match="No updates provided"):
            await repo.update("theme_x", {})

    @pytest.mark.asyncio
    async def test_bad_lifecycle_raises(
        self, mock_database: AsyncMock,
    ) -> None:
        repo = ThemeRepository(mock_database)

        with pytest.raises(ValueError, match="Invalid lifecycle_stage"):
            await repo.update("theme_x", {"lifecycle_stage": "dead"})

    @pytest.mark.asyncio
    async def test_not_found_raises(
        self, mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchrow.return_value = None
        repo = ThemeRepository(mock_database)

        with pytest.raises(ValueError, match="not found"):
            await repo.update("theme_missing", {"name": "gone"})

    @pytest.mark.asyncio
    async def test_jsonb_serialization(
        self,
        mock_database: AsyncMock,
        sample_db_row: dict,
    ) -> None:
        """JSONB fields are json.dumps'd before passing to DB."""
        mock_database.fetchrow.return_value = sample_db_row
        repo = ThemeRepository(mock_database)

        entities = [{"type": "COMPANY", "normalized": "AMD"}]
        await repo.update(
            "theme_a1b2c3d4e5f6", {"top_entities": entities},
        )

        args = mock_database.fetchrow.call_args[0]
        # The JSONB param should be a string
        assert args[1] == json.dumps(entities)

    @pytest.mark.asyncio
    async def test_centroid_excluded(
        self, mock_database: AsyncMock,
    ) -> None:
        """centroid is not in the updatable allowlist."""
        repo = ThemeRepository(mock_database)

        with pytest.raises(ValueError, match="Invalid fields.*centroid"):
            await repo.update("theme_x", {"centroid": [0.0] * 768})


class TestUpdateCentroid:
    """Test ThemeRepository.update_centroid()."""

    @pytest.mark.asyncio
    async def test_success(self, mock_database: AsyncMock) -> None:
        mock_database.execute.return_value = "UPDATE 1"
        repo = ThemeRepository(mock_database)
        centroid = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        await repo.update_centroid("theme_abc", centroid)

        args = mock_database.execute.call_args[0]
        sql = args[0]
        assert "SET centroid = $2" in sql
        assert "WHERE theme_id = $1" in sql
        assert args[1] == "theme_abc"
        assert args[2] == _centroid_to_pgvector(centroid)

    @pytest.mark.asyncio
    async def test_not_found_raises(self, mock_database: AsyncMock) -> None:
        mock_database.execute.return_value = "UPDATE 0"
        repo = ThemeRepository(mock_database)

        with pytest.raises(ValueError, match="not found"):
            await repo.update_centroid("theme_missing", np.zeros(768))

    @pytest.mark.asyncio
    async def test_pgvector_formatting(
        self, mock_database: AsyncMock,
    ) -> None:
        mock_database.execute.return_value = "UPDATE 1"
        repo = ThemeRepository(mock_database)
        centroid = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        await repo.update_centroid("theme_abc", centroid)

        pgvec = mock_database.execute.call_args[0][2]
        assert pgvec == "[1.0,2.0,3.0]"


class TestDelete:
    """Test ThemeRepository.delete()."""

    @pytest.mark.asyncio
    async def test_found_returns_true(
        self, mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchval.return_value = "theme_abc"
        repo = ThemeRepository(mock_database)

        result = await repo.delete("theme_abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_not_found_returns_false(
        self, mock_database: AsyncMock,
    ) -> None:
        mock_database.fetchval.return_value = None
        repo = ThemeRepository(mock_database)

        result = await repo.delete("theme_nonexistent")
        assert result is False


class TestRowToTheme:
    """Test _row_to_theme helper."""

    def test_pgvector_string_parsing(self, sample_db_row: dict) -> None:
        """pgvector string "[0.1,0.2,...]" is parsed to ndarray."""
        theme = _row_to_theme(sample_db_row)
        assert isinstance(theme.centroid, np.ndarray)
        assert theme.centroid.dtype == np.float32
        assert len(theme.centroid) == 768

    def test_jsonb_string_parsing(self, sample_db_row: dict) -> None:
        """JSONB string fields are parsed to Python objects."""
        theme = _row_to_theme(sample_db_row)
        assert isinstance(theme.top_entities, list)
        assert theme.top_entities[0]["type"] == "COMPANY"
        assert isinstance(theme.metadata, dict)

    def test_already_parsed_jsonb(self, sample_db_row: dict) -> None:
        """Already-parsed JSONB (list/dict) passes through."""
        sample_db_row["top_entities"] = [{"type": "TICKER"}]
        sample_db_row["metadata"] = {"key": "val"}
        theme = _row_to_theme(sample_db_row)
        assert theme.top_entities == [{"type": "TICKER"}]
        assert theme.metadata == {"key": "val"}

    def test_text_array_passthrough(self, sample_db_row: dict) -> None:
        """TEXT[] columns pass through as Python lists."""
        theme = _row_to_theme(sample_db_row)
        assert theme.top_keywords == ["gpu", "nvidia", "architecture"]
        assert theme.top_tickers == ["NVDA", "AMD"]

    def test_defaults_for_missing_fields(self) -> None:
        """Missing optional fields get defaults."""
        row = {
            "theme_id": "theme_min",
            "name": "minimal",
            "centroid": "[0.0,0.0,0.0]",
            "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
        }
        theme = _row_to_theme(row)
        assert theme.top_keywords == []
        assert theme.top_tickers == []
        assert theme.top_entities == []
        assert theme.metadata == {}
        assert theme.document_count == 0
        assert theme.lifecycle_stage == "emerging"
        assert theme.description is None


class TestParseCentroid:
    """Test _parse_centroid helper."""

    def test_string_input(self) -> None:
        result = _parse_centroid("[0.1,0.2,0.3]")
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3], atol=1e-6)
        assert result.dtype == np.float32

    def test_list_input(self) -> None:
        result = _parse_centroid([0.1, 0.2, 0.3])
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3], atol=1e-6)

    def test_ndarray_input(self) -> None:
        arr = np.array([0.1, 0.2], dtype=np.float64)
        result = _parse_centroid(arr)
        assert result.dtype == np.float32

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse centroid"):
            _parse_centroid(42)


class TestCentroidToPgvector:
    """Test _centroid_to_pgvector helper."""

    def test_basic(self) -> None:
        centroid = np.array([1.0, 2.0, 3.0])
        result = _centroid_to_pgvector(centroid)
        assert result == "[1.0,2.0,3.0]"

    def test_float32_precision(self) -> None:
        centroid = np.array([0.123456789], dtype=np.float32)
        result = _centroid_to_pgvector(centroid)
        assert result.startswith("[0.123")
        assert result.endswith("]")
