"""Tests for platform adapters."""

import pytest

from src.ingestion.mock_adapter import MockAdapter, create_mock_adapters
from src.ingestion.schemas import NormalizedDocument, Platform


class TestMockAdapter:
    """Tests for MockAdapter."""

    @pytest.mark.asyncio
    async def test_fetch_returns_documents(self):
        """Mock adapter should return documents."""
        adapter = MockAdapter(
            platform=Platform.TWITTER,
            documents_per_fetch=5,
        )

        docs = []
        async for doc in adapter.fetch():
            docs.append(doc)

        assert len(docs) == 5
        assert all(isinstance(d, NormalizedDocument) for d in docs)
        assert all(d.platform == Platform.TWITTER for d in docs)

    @pytest.mark.asyncio
    async def test_documents_have_required_fields(self):
        """Mock documents should have all required fields."""
        adapter = MockAdapter(documents_per_fetch=1)

        async for doc in adapter.fetch():
            assert doc.id
            assert doc.platform
            assert doc.timestamp
            assert doc.author_id
            assert doc.author_name
            assert doc.content
            assert len(doc.tickers_mentioned) > 0

    @pytest.mark.asyncio
    async def test_different_platforms(self):
        """Mock adapters should respect platform setting."""
        for platform in Platform:
            adapter = MockAdapter(platform=platform, documents_per_fetch=1)
            async for doc in adapter.fetch():
                assert doc.platform == platform

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Mock adapter health check should pass."""
        adapter = MockAdapter()
        assert await adapter.health_check() is True

    @pytest.mark.asyncio
    async def test_spam_content_included(self):
        """Mock adapter should include spam-like content when enabled."""
        adapter = MockAdapter(
            documents_per_fetch=100,
            include_spam=True,
        )

        spam_count = 0
        async for doc in adapter.fetch():
            if "discord" in doc.content.lower() or "join" in doc.content.lower():
                spam_count += 1

        # Should have some spam content (roughly 10%)
        assert spam_count > 0


class TestCreateMockAdapters:
    """Tests for create_mock_adapters factory."""

    def test_creates_all_platforms(self):
        """Factory should create adapters for all platforms."""
        adapters = create_mock_adapters()

        assert len(adapters) == 4
        assert Platform.TWITTER in adapters
        assert Platform.REDDIT in adapters
        assert Platform.SUBSTACK in adapters
        assert Platform.NEWS in adapters

    def test_custom_documents_per_fetch(self):
        """Factory should respect documents_per_fetch parameter."""
        adapters = create_mock_adapters(documents_per_fetch=20)

        # Substack gets half the documents
        assert adapters[Platform.TWITTER]._documents_per_fetch == 20
        assert adapters[Platform.SUBSTACK]._documents_per_fetch == 10
