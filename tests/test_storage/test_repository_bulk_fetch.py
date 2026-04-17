"""Tests for bulk document fetch repository behavior."""

import sys
import types
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

if "asyncpg" not in sys.modules:
    asyncpg_stub = types.ModuleType("asyncpg")
    asyncpg_stub.Connection = object
    asyncpg_stub.Pool = object
    asyncpg_stub.Record = dict
    asyncpg_stub.create_pool = AsyncMock()
    sys.modules["asyncpg"] = asyncpg_stub

from src.storage.repository import DocumentRepository


@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    return db


@pytest.fixture
def repo(mock_db):
    return DocumentRepository(mock_db)


def row(doc_id: str) -> dict[str, object]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return {
        "id": doc_id,
        "platform": "news",
        "url": f"https://example.com/{doc_id}",
        "timestamp": timestamp,
        "fetched_at": timestamp,
        "author_id": "author_1",
        "author_name": "Author",
        "author_followers": 123,
        "author_verified": False,
        "content": f"Content for {doc_id}",
        "content_type": "post",
        "title": f"Title for {doc_id}",
        "engagement": {"likes": 1, "shares": 2, "comments": 3, "views": 4},
        "tickers": ["AAPL"],
        "entities_mentioned": [],
        "keywords_extracted": [],
        "events_extracted": [],
        "urls_mentioned": [],
        "spam_score": 0.1,
        "bot_probability": 0.2,
        "authority_score": 0.3,
        "embedding": None,
        "embedding_minilm": None,
        "sentiment": None,
        "theme_ids": [],
        "raw_data": {},
    }


@pytest.mark.asyncio
async def test_get_by_ids_uses_any_array_and_preserves_order(repo, mock_db):
    mock_db.fetch.return_value = [row("doc_2"), row("doc_1")]

    result = await repo.get_by_ids(["doc_1", "missing", "doc_2"])

    mock_db.fetch.assert_awaited_once()
    sql, ids = mock_db.fetch.await_args.args
    assert sql == "SELECT * FROM documents WHERE id = ANY($1::text[])"
    assert ids == ["doc_1", "missing", "doc_2"]
    assert [doc.id for doc in result] == ["doc_1", "doc_2"]


@pytest.mark.asyncio
async def test_get_by_ids_empty_returns_empty_without_fetch(repo, mock_db):
    result = await repo.get_by_ids([])

    assert result == []
    mock_db.fetch.assert_not_awaited()
