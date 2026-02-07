"""Tests for AlertRepository with mocked Database."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from src.alerts.repository import AlertRepository, _row_to_alert
from src.alerts.schemas import Alert


@pytest.fixture
def mock_db():
    db = AsyncMock()
    return db


@pytest.fixture
def repo(mock_db):
    return AlertRepository(mock_db)


def _make_db_row(**overrides):
    """Create a mock asyncpg Record as a dict."""
    row = {
        "alert_id": "test-alert-id",
        "theme_id": "theme_abc",
        "trigger_type": "volume_surge",
        "severity": "warning",
        "title": "Volume surge",
        "message": "Z-score is 3.5",
        "trigger_data": {"volume_zscore": 3.5},
        "acknowledged": False,
        "created_at": datetime(2026, 2, 7, 10, 0, 0, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return row


class TestRowToAlert:
    """Test the module-level _row_to_alert helper."""

    def test_basic_conversion(self):
        row = _make_db_row()
        alert = _row_to_alert(row)
        assert alert.alert_id == "test-alert-id"
        assert alert.trigger_type == "volume_surge"
        assert alert.trigger_data == {"volume_zscore": 3.5}

    def test_trigger_data_as_string(self):
        row = _make_db_row(trigger_data='{"key": "val"}')
        alert = _row_to_alert(row)
        assert alert.trigger_data == {"key": "val"}


class TestCreate:
    """Test alert creation."""

    @pytest.mark.asyncio
    async def test_create_calls_db(self, repo, mock_db):
        mock_db.fetchrow.return_value = _make_db_row()
        alert = Alert(
            theme_id="theme_abc",
            trigger_type="volume_surge",
            severity="warning",
            title="Test",
            message="Test msg",
        )
        result = await repo.create(alert)
        assert result.alert_id == "test-alert-id"
        mock_db.fetchrow.assert_called_once()


class TestCreateBatch:
    """Test batch alert creation."""

    @pytest.mark.asyncio
    async def test_batch_creates_all(self, repo, mock_db):
        mock_db.fetchrow.return_value = _make_db_row()
        alerts = [
            Alert(
                theme_id=f"t{i}",
                trigger_type="new_theme",
                severity="info",
                title=f"Alert {i}",
                message=f"Msg {i}",
            )
            for i in range(3)
        ]
        result = await repo.create_batch(alerts)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_batch_handles_individual_errors(self, repo, mock_db):
        call_count = 0

        async def side_effect(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("DB error on second insert")
            return _make_db_row()

        mock_db.fetchrow.side_effect = side_effect
        alerts = [
            Alert(
                theme_id=f"t{i}",
                trigger_type="new_theme",
                severity="info",
                title=f"Alert {i}",
                message=f"Msg {i}",
            )
            for i in range(3)
        ]
        result = await repo.create_batch(alerts)
        assert len(result) == 2  # 1 failed, 2 succeeded


class TestGetRecent:
    """Test filtered alert retrieval."""

    @pytest.mark.asyncio
    async def test_no_filters(self, repo, mock_db):
        mock_db.fetch.return_value = [_make_db_row()]
        result = await repo.get_recent()
        assert len(result) == 1
        mock_db.fetch.assert_called_once()
        # Check SQL has no WHERE (only LIMIT/OFFSET)
        sql = mock_db.fetch.call_args[0][0]
        assert "WHERE" not in sql

    @pytest.mark.asyncio
    async def test_severity_filter(self, repo, mock_db):
        mock_db.fetch.return_value = [_make_db_row()]
        await repo.get_recent(severity="critical")
        sql = mock_db.fetch.call_args[0][0]
        assert "severity = $1" in sql

    @pytest.mark.asyncio
    async def test_multiple_filters(self, repo, mock_db):
        mock_db.fetch.return_value = []
        await repo.get_recent(
            severity="warning",
            trigger_type="volume_surge",
            theme_id="t1",
        )
        sql = mock_db.fetch.call_args[0][0]
        assert "severity = $1" in sql
        assert "trigger_type = $2" in sql
        assert "theme_id = $3" in sql

    @pytest.mark.asyncio
    async def test_acknowledged_filter(self, repo, mock_db):
        mock_db.fetch.return_value = []
        await repo.get_recent(acknowledged=False)
        sql = mock_db.fetch.call_args[0][0]
        assert "acknowledged = $1" in sql

    @pytest.mark.asyncio
    async def test_pagination_params(self, repo, mock_db):
        mock_db.fetch.return_value = []
        await repo.get_recent(limit=10, offset=20)
        args = mock_db.fetch.call_args[0]
        # limit and offset are the last two params
        assert args[-2] == 10
        assert args[-1] == 20


class TestCountTodayBySeverity:
    """Test rate limit counting."""

    @pytest.mark.asyncio
    async def test_returns_count(self, repo, mock_db):
        mock_db.fetchval.return_value = 7
        result = await repo.count_today_by_severity("critical")
        assert result == 7

    @pytest.mark.asyncio
    async def test_returns_zero_for_null(self, repo, mock_db):
        mock_db.fetchval.return_value = None
        result = await repo.count_today_by_severity("critical")
        assert result == 0


class TestAcknowledge:
    """Test alert acknowledgement."""

    @pytest.mark.asyncio
    async def test_acknowledge_success(self, repo, mock_db):
        mock_db.fetchval.return_value = "test-id"
        result = await repo.acknowledge("test-id")
        assert result is True

    @pytest.mark.asyncio
    async def test_acknowledge_not_found(self, repo, mock_db):
        mock_db.fetchval.return_value = None
        result = await repo.acknowledge("nonexistent")
        assert result is False


class TestGetById:
    """Test single alert retrieval."""

    @pytest.mark.asyncio
    async def test_found(self, repo, mock_db):
        mock_db.fetchrow.return_value = _make_db_row()
        result = await repo.get_by_id("test-alert-id")
        assert result.alert_id == "test-alert-id"

    @pytest.mark.asyncio
    async def test_not_found(self, repo, mock_db):
        mock_db.fetchrow.return_value = None
        result = await repo.get_by_id("nonexistent")
        assert result is None
