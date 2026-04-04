"""Tests for FilingProvider interface and SEC policy."""

from datetime import date

import pytest

from src.filing.config import FilingConfig
from src.filing.provider import FilingProvider, SECRateLimiter
from src.filing.schemas import FilingIdentity, FilingResult
from src.filing.sec_policy import SECPolicy


class TestSECPolicy:
    """SEC policy configuration."""

    def test_defaults(self) -> None:
        policy = SECPolicy()
        assert policy.rate_limit_per_second == 8
        assert policy.max_retries == 3
        assert "sec.gov" in policy.base_url

    def test_user_agent_present(self) -> None:
        policy = SECPolicy()
        assert len(policy.user_agent) > 0
        assert "news-tracker" in policy.user_agent

    def test_headers_include_user_agent(self) -> None:
        policy = SECPolicy()
        headers = policy.headers
        assert "User-Agent" in headers
        assert headers["User-Agent"] == policy.user_agent

    def test_archives_url(self) -> None:
        policy = SECPolicy()
        assert "Archives/edgar" in policy.archives_url


class TestSECRateLimiter:
    """SEC rate limiter."""

    async def test_acquire_succeeds(self) -> None:
        limiter = SECRateLimiter(requests_per_second=10)
        await limiter.acquire()

    async def test_multiple_acquires_within_limit(self) -> None:
        limiter = SECRateLimiter(requests_per_second=10)
        for _ in range(5):
            await limiter.acquire()


class TestFilingConfig:
    """Filing lane configuration."""

    def test_defaults(self) -> None:
        config = FilingConfig()
        assert config.primary_provider == "edgartools"
        assert config.fallback_enabled is True

    def test_filing_type_list(self) -> None:
        config = FilingConfig()
        types = config.filing_type_list
        assert "10-K" in types
        assert "10-Q" in types
        assert "8-K" in types

    def test_custom_filing_types(self) -> None:
        config = FilingConfig(filing_types="10-K,20-F")
        assert config.filing_type_list == ["10-K", "20-F"]


class _StubProvider(FilingProvider):
    """Minimal concrete implementation for testing the ABC."""

    @property
    def name(self) -> str:
        return "stub"

    async def fetch_filing(self, accession_number: str) -> FilingResult:
        return FilingResult(
            identity=FilingIdentity(
                cik="001",
                accession_number=accession_number,
                filing_type="10-K",
                filed_date=date(2024, 1, 1),
            ),
            provider=self.name,
        )

    async def search_filings(self, **kwargs) -> list[FilingIdentity]:
        return []


class TestFilingProviderContract:
    """Verify the FilingProvider ABC can be implemented."""

    def test_stub_instantiates(self) -> None:
        provider = _StubProvider()
        assert provider.name == "stub"

    def test_stub_has_policy(self) -> None:
        provider = _StubProvider()
        assert isinstance(provider.policy, SECPolicy)

    def test_custom_policy(self) -> None:
        policy = SECPolicy(rate_limit_per_second=5)
        provider = _StubProvider(policy=policy)
        assert provider.policy.rate_limit_per_second == 5

    async def test_fetch_returns_filing_result(self) -> None:
        provider = _StubProvider()
        result = await provider.fetch_filing("acc-001")
        assert isinstance(result, FilingResult)
        assert result.provider == "stub"
        assert result.identity.accession_number == "acc-001"

    async def test_search_returns_list(self) -> None:
        provider = _StubProvider()
        results = await provider.search_filings(cik="001")
        assert isinstance(results, list)

    async def test_rate_limiter_accessible(self) -> None:
        provider = _StubProvider()
        await provider._acquire_rate_limit()
