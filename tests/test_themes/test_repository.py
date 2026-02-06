"""Tests for ThemeRepository CRUD, vector search, and metrics operations."""

import json
import time
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, call, patch

import numpy as np
import pytest

from src.themes.repository import (
    ThemeRepository,
    _centroid_to_pgvector,
    _parse_centroid,
    _row_to_metrics,
    _row_to_theme,
)
from src.themes.schemas import Theme, ThemeMetrics


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


# ── Vector Search Tests ──────────────────────────────────────


class TestFindSimilar:
    """Test ThemeRepository.find_similar()."""

    @pytest.mark.asyncio
    async def test_returns_themes_with_similarity(
        self,
        mock_database: AsyncMock,
        sample_similarity_row: dict,
    ) -> None:
        """find_similar returns (Theme, similarity) tuples."""
        mock_database.fetch.return_value = [sample_similarity_row]
        repo = ThemeRepository(mock_database)
        query = np.random.default_rng(0).standard_normal(768).astype(np.float32)

        results = await repo.find_similar(query, limit=5, threshold=0.5)

        assert len(results) == 1
        theme, score = results[0]
        assert isinstance(theme, Theme)
        assert theme.theme_id == "theme_a1b2c3d4e5f6"
        assert score == pytest.approx(0.92)

    @pytest.mark.asyncio
    async def test_sql_uses_cosine_distance(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """SQL uses the pgvector <=> cosine distance operator."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)
        query = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        await repo.find_similar(query)

        sql = mock_database.fetch.call_args[0][0]
        assert "centroid <=> $1" in sql
        assert "1 - (centroid <=> $1)" in sql
        assert "ORDER BY centroid <=> $1" in sql
        assert "LIMIT" in sql

    @pytest.mark.asyncio
    async def test_passes_threshold_and_limit(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Threshold and limit are passed as query params."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)
        query = np.array([0.1], dtype=np.float32)

        await repo.find_similar(query, limit=20, threshold=0.8)

        args = mock_database.fetch.call_args[0]
        # $1 = centroid string, $2 = threshold, $3 = limit
        assert args[2] == 0.8
        assert args[3] == 20

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_database: AsyncMock) -> None:
        """No themes above threshold returns empty list."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)
        query = np.zeros(768, dtype=np.float32)

        results = await repo.find_similar(query)
        assert results == []

    @pytest.mark.asyncio
    async def test_centroid_format(self, mock_database: AsyncMock) -> None:
        """Query centroid is passed as pgvector string."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)
        query = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        await repo.find_similar(query)

        centroid_param = mock_database.fetch.call_args[0][1]
        assert centroid_param == "[1.0,2.0,3.0]"


class TestGetCentroidsBatch:
    """Test ThemeRepository.get_centroids_batch()."""

    @pytest.mark.asyncio
    async def test_fetches_from_db(
        self,
        mock_database: AsyncMock,
        sample_centroid: np.ndarray,
    ) -> None:
        """Uncached IDs are fetched from the database."""
        centroid_str = _centroid_to_pgvector(sample_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": centroid_str},
        ]
        repo = ThemeRepository(mock_database)

        result = await repo.get_centroids_batch(["theme_abc"])

        assert "theme_abc" in result
        np.testing.assert_allclose(
            result["theme_abc"], sample_centroid, atol=1e-5,
        )
        # Verify single-query fetch with ANY
        sql = mock_database.fetch.call_args[0][0]
        assert "theme_id = ANY($1)" in sql

    @pytest.mark.asyncio
    async def test_cache_hit_skips_db(
        self,
        mock_database: AsyncMock,
        sample_centroid: np.ndarray,
    ) -> None:
        """Cached centroids don't trigger a DB query."""
        centroid_str = _centroid_to_pgvector(sample_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": centroid_str},
        ]
        repo = ThemeRepository(mock_database)

        # First call populates cache
        await repo.get_centroids_batch(["theme_abc"])
        assert mock_database.fetch.call_count == 1

        # Second call should hit cache
        result = await repo.get_centroids_batch(["theme_abc"])
        assert mock_database.fetch.call_count == 1  # No new DB call
        assert "theme_abc" in result

    @pytest.mark.asyncio
    async def test_partial_cache_hit(
        self,
        mock_database: AsyncMock,
        sample_centroid: np.ndarray,
    ) -> None:
        """Mix of cached and uncached IDs fetches only uncached."""
        centroid_str = _centroid_to_pgvector(sample_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": centroid_str},
        ]
        repo = ThemeRepository(mock_database)

        # Populate cache for theme_abc
        await repo.get_centroids_batch(["theme_abc"])

        # Now request both cached + uncached
        other = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_xyz", "centroid": _centroid_to_pgvector(other)},
        ]
        result = await repo.get_centroids_batch(["theme_abc", "theme_xyz"])

        assert "theme_abc" in result
        assert "theme_xyz" in result
        # Second fetch should only request theme_xyz
        ids_arg = mock_database.fetch.call_args[0][1]
        assert ids_arg == ["theme_xyz"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_centroid_update(
        self,
        mock_database: AsyncMock,
        sample_centroid: np.ndarray,
    ) -> None:
        """update_centroid() invalidates cached entry."""
        centroid_str = _centroid_to_pgvector(sample_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": centroid_str},
        ]
        mock_database.execute.return_value = "UPDATE 1"
        repo = ThemeRepository(mock_database)

        # Populate cache
        await repo.get_centroids_batch(["theme_abc"])
        assert mock_database.fetch.call_count == 1

        # Update centroid should invalidate cache
        new_centroid = np.ones(768, dtype=np.float32)
        await repo.update_centroid("theme_abc", new_centroid)

        # Next batch fetch should go to DB
        new_str = _centroid_to_pgvector(new_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": new_str},
        ]
        await repo.get_centroids_batch(["theme_abc"])
        assert mock_database.fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_delete(
        self,
        mock_database: AsyncMock,
        sample_centroid: np.ndarray,
    ) -> None:
        """delete() invalidates cached centroid."""
        centroid_str = _centroid_to_pgvector(sample_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": centroid_str},
        ]
        mock_database.fetchval.return_value = "theme_abc"
        repo = ThemeRepository(mock_database)

        # Populate cache
        await repo.get_centroids_batch(["theme_abc"])

        # Delete invalidates cache
        await repo.delete("theme_abc")

        # Next fetch should go to DB
        mock_database.fetch.return_value = []
        result = await repo.get_centroids_batch(["theme_abc"])
        assert result == {}
        assert mock_database.fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_ttl_expiry(
        self,
        mock_database: AsyncMock,
        sample_centroid: np.ndarray,
    ) -> None:
        """Expired cache entries are re-fetched from DB."""
        centroid_str = _centroid_to_pgvector(sample_centroid)
        mock_database.fetch.return_value = [
            {"theme_id": "theme_abc", "centroid": centroid_str},
        ]
        # Use a very short TTL for testing
        repo = ThemeRepository(mock_database, centroid_cache_ttl=0.01)

        await repo.get_centroids_batch(["theme_abc"])
        assert mock_database.fetch.call_count == 1

        # Wait for TTL to expire
        time.sleep(0.02)

        await repo.get_centroids_batch(["theme_abc"])
        assert mock_database.fetch.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_input(self, mock_database: AsyncMock) -> None:
        """Empty theme_ids returns empty dict without DB query."""
        repo = ThemeRepository(mock_database)

        result = await repo.get_centroids_batch([])

        assert result == {}
        mock_database.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_themes_omitted(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Theme IDs not in DB are silently omitted from result."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)

        result = await repo.get_centroids_batch(["theme_missing"])
        assert result == {}


# ── Metrics Time Series Tests ────────────────────────────────


class TestAddMetrics:
    """Test ThemeRepository.add_metrics()."""

    @pytest.mark.asyncio
    async def test_upsert_sql(
        self,
        mock_database: AsyncMock,
        sample_metrics: ThemeMetrics,
    ) -> None:
        """add_metrics uses INSERT ... ON CONFLICT DO UPDATE."""
        mock_database.execute.return_value = "INSERT 0 1"
        repo = ThemeRepository(mock_database)

        await repo.add_metrics(sample_metrics)

        args = mock_database.execute.call_args[0]
        sql = args[0]
        assert "INSERT INTO theme_metrics" in sql
        assert "ON CONFLICT (theme_id, date) DO UPDATE" in sql

    @pytest.mark.asyncio
    async def test_params_ordering(
        self,
        mock_database: AsyncMock,
        sample_metrics: ThemeMetrics,
    ) -> None:
        """Parameters are passed in the correct order."""
        mock_database.execute.return_value = "INSERT 0 1"
        repo = ThemeRepository(mock_database)

        await repo.add_metrics(sample_metrics)

        args = mock_database.execute.call_args[0]
        assert args[1] == "theme_a1b2c3d4e5f6"  # theme_id
        assert args[2] == date(2025, 6, 15)       # date
        assert args[3] == 42                       # document_count
        assert args[4] == pytest.approx(0.35)      # sentiment_score
        assert args[5] == pytest.approx(1.8)       # volume_zscore
        assert args[6] == pytest.approx(0.12)      # velocity
        assert args[7] == pytest.approx(0.03)      # acceleration
        assert args[8] == pytest.approx(0.65)      # avg_authority
        assert args[9] == pytest.approx(0.72)      # bullish_ratio

    @pytest.mark.asyncio
    async def test_nullable_fields(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Optional fields pass None to the DB."""
        mock_database.execute.return_value = "INSERT 0 1"
        repo = ThemeRepository(mock_database)
        metrics = ThemeMetrics(
            theme_id="theme_sparse",
            date=date(2025, 6, 15),
            document_count=10,
        )

        await repo.add_metrics(metrics)

        args = mock_database.execute.call_args[0]
        # sentiment_score through bullish_ratio should all be None
        for i in range(4, 10):
            assert args[i] is None


class TestGetMetricsRange:
    """Test ThemeRepository.get_metrics_range()."""

    @pytest.mark.asyncio
    async def test_sql_and_params(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Correct SQL with date range filter."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)

        await repo.get_metrics_range(
            "theme_abc",
            start=date(2025, 6, 1),
            end=date(2025, 6, 30),
        )

        args = mock_database.fetch.call_args[0]
        sql = args[0]
        assert "theme_metrics" in sql
        assert "date >= $2" in sql
        assert "date <= $3" in sql
        assert "ORDER BY date ASC" in sql
        assert args[1] == "theme_abc"
        assert args[2] == date(2025, 6, 1)
        assert args[3] == date(2025, 6, 30)

    @pytest.mark.asyncio
    async def test_returns_ordered_metrics(
        self,
        mock_database: AsyncMock,
        sample_metrics_row: dict,
    ) -> None:
        """Rows are converted to ThemeMetrics objects."""
        row2 = {**sample_metrics_row, "date": date(2025, 6, 16)}
        mock_database.fetch.return_value = [sample_metrics_row, row2]
        repo = ThemeRepository(mock_database)

        results = await repo.get_metrics_range(
            "theme_a1b2c3d4e5f6",
            start=date(2025, 6, 15),
            end=date(2025, 6, 16),
        )

        assert len(results) == 2
        assert all(isinstance(m, ThemeMetrics) for m in results)
        assert results[0].date == date(2025, 6, 15)
        assert results[1].date == date(2025, 6, 16)

    @pytest.mark.asyncio
    async def test_empty_range(self, mock_database: AsyncMock) -> None:
        """No metrics in range returns empty list."""
        mock_database.fetch.return_value = []
        repo = ThemeRepository(mock_database)

        results = await repo.get_metrics_range(
            "theme_abc",
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
        )
        assert results == []


class TestRowToMetrics:
    """Test _row_to_metrics helper."""

    def test_full_row(self, sample_metrics_row: dict) -> None:
        metrics = _row_to_metrics(sample_metrics_row)
        assert isinstance(metrics, ThemeMetrics)
        assert metrics.theme_id == "theme_a1b2c3d4e5f6"
        assert metrics.date == date(2025, 6, 15)
        assert metrics.document_count == 42
        assert metrics.sentiment_score == pytest.approx(0.35)
        assert metrics.bullish_ratio == pytest.approx(0.72)

    def test_nullable_fields(self) -> None:
        row = {
            "theme_id": "theme_sparse",
            "date": date(2025, 6, 1),
            "document_count": 5,
            "sentiment_score": None,
            "volume_zscore": None,
            "velocity": None,
            "acceleration": None,
            "avg_authority": None,
            "bullish_ratio": None,
        }
        metrics = _row_to_metrics(row)
        assert metrics.sentiment_score is None
        assert metrics.velocity is None
        assert metrics.bullish_ratio is None
