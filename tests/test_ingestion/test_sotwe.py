"""Tests for Twitter xui guardrails and adapter behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.config.twitter_accounts import DEFAULT_USERNAMES, get_default_usernames, parse_usernames
from src.ingestion.schemas import Platform
from src.ingestion.twitter_adapter import TwitterAdapter, XuiInvocationResult


def _settings(**overrides):
    defaults = {
        "twitter_bearer_token": None,
        "twitter_xui_enabled": True,
        "twitter_xui_command": "xui",
        "twitter_xui_config_path": None,
        "twitter_xui_profile": "default",
        "twitter_xui_profile_dir": None,
        "twitter_xui_usernames": "nvidia,amd",
        "twitter_xui_poll_min_seconds": 120,
        "twitter_xui_poll_max_seconds": 300,
        "twitter_xui_cycle_jitter_ratio": 0.25,
        "twitter_xui_limit_per_source": 50,
        "twitter_xui_scroll_pause_min_ms": 1400,
        "twitter_xui_scroll_pause_max_ms": 3200,
        "twitter_xui_max_scroll_rounds": 4,
        "twitter_xui_max_page_loads": 2,
        "twitter_xui_timeout_ms": 90000,
        "twitter_xui_shuffle_sources": True,
        "twitter_xui_source_cooldown_cycles": 2,
        "twitter_xui_source_pause_min_seconds": 0.8,
        "twitter_xui_source_pause_max_seconds": 2.5,
        "twitter_xui_block_backoff_initial_seconds": 300,
        "twitter_xui_block_backoff_max_seconds": 3600,
        "twitter_xui_block_circuit_threshold": 3,
        "twitter_xui_block_circuit_open_seconds": 7200,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestTwitterAccounts:
    def test_default_usernames_not_empty(self):
        usernames = get_default_usernames()
        assert len(usernames) > 0
        assert "SemiAnalysis" in usernames

    def test_default_usernames_returns_copy(self):
        usernames1 = get_default_usernames()
        usernames2 = get_default_usernames()
        usernames1.append("test")
        assert "test" not in usernames2

    def test_parse_usernames_with_valid_string(self):
        result = parse_usernames("user1,user2,user3")
        assert result == ["user1", "user2", "user3"]

    def test_parse_usernames_strips_whitespace(self):
        result = parse_usernames(" user1 , user2 , user3 ")
        assert result == ["user1", "user2", "user3"]

    def test_parse_usernames_strips_at_prefix(self):
        result = parse_usernames("@user1,@user2,user3")
        assert result == ["user1", "user2", "user3"]

    def test_parse_usernames_with_none_returns_defaults(self):
        result = parse_usernames(None)
        assert result == DEFAULT_USERNAMES

    def test_parse_usernames_with_empty_string_returns_defaults(self):
        result = parse_usernames("")
        assert result == DEFAULT_USERNAMES

    def test_parse_usernames_filters_empty(self):
        result = parse_usernames("user1,,user2,")
        assert result == ["user1", "user2"]


class TestTwitterAdapterXui:
    def test_adapter_uses_xui_when_no_token(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings(twitter_bearer_token=None)

            adapter = TwitterAdapter()

            assert adapter._bearer_token is None
            assert adapter._xui.enabled is True
            assert adapter._xui.usernames == ("nvidia", "amd")

    def test_adapter_transform_xui_tweet(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings()
            adapter = TwitterAdapter()

        raw = {
            "source": "xui",
            "tweet": {
                "tweet_id": "123456789",
                "text": "NVIDIA stock up 5% today $NVDA bullish",
                "created_at": "2024-01-15T10:30:00Z",
                "author_handle": "SemiAnalysis",
                "author_display_name": "SemiAnalysis",
                "author_verified": True,
                "likes": 100,
                "retweets": 50,
                "replies": 25,
                "views": 1000,
                "source_id": "user:semianalysis",
            },
            "username": "SemiAnalysis",
        }

        doc = adapter._transform(raw)
        assert doc is not None
        assert doc.id == "twitter_123456789"
        assert doc.platform == Platform.TWITTER
        assert "NVDA" in doc.tickers_mentioned
        assert doc.engagement.likes == 100
        assert doc.engagement.shares == 50
        assert doc.engagement.views == 1000
        assert doc.author_name == "SemiAnalysis"
        assert doc.author_id == "SemiAnalysis"
        assert doc.raw_data["source"] == "xui"
        assert doc.raw_data["ingestion_method"] == "xui"
        assert doc.url == "https://x.com/SemiAnalysis/status/123456789"

    def test_next_poll_delay_respects_bounds(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings(
                twitter_xui_poll_min_seconds=120,
                twitter_xui_poll_max_seconds=300,
            )
            adapter = TwitterAdapter()

        for _ in range(50):
            delay = adapter.next_poll_delay_seconds(default_interval_seconds=3600)
            assert 120 <= delay <= 300

    def test_block_backoff_and_circuit_breaker(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings(
                twitter_xui_block_backoff_initial_seconds=300,
                twitter_xui_block_backoff_max_seconds=3600,
                twitter_xui_block_circuit_threshold=3,
            )
            adapter = TwitterAdapter()

        blocked = XuiInvocationResult(
            return_code=2,
            stdout="",
            stderr="403 challenge",
            payload={},
        )

        adapter._register_xui_block("challenge", blocked)
        assert adapter._xui_state.consecutive_block_events == 1
        assert adapter._xui_state.current_backoff_seconds == 300
        assert adapter._xui_state.circuit_open_until is None

        adapter._register_xui_block("challenge", blocked)
        assert adapter._xui_state.current_backoff_seconds == 600
        assert adapter._xui_state.circuit_open_until is None

        adapter._register_xui_block("challenge", blocked)
        assert adapter._xui_state.current_backoff_seconds == 1200
        assert adapter._xui_state.circuit_open_until is not None
        assert adapter._xui_state.circuit_open_until > datetime.now(UTC)

    @pytest.mark.asyncio
    async def test_fetch_raw_falls_back_to_api_when_xui_unsuccessful(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings(twitter_bearer_token="api_token")
            adapter = TwitterAdapter(bearer_token="api_token")

        adapter._collect_xui_items = AsyncMock(return_value=([], False))

        async def _fake_api():
            yield {
                "source": "twitter_api",
                "tweet": {
                    "id": "111",
                    "text": "$AMD up",
                    "created_at": "2024-01-15T10:30:00Z",
                    "author_id": "42",
                    "public_metrics": {},
                },
                "author": {"username": "AMD"},
            }

        adapter._fetch_twitter_api = _fake_api

        items = [item async for item in adapter._fetch_raw()]
        assert len(items) == 1
        assert items[0]["source"] == "twitter_api"

    def test_detect_xui_block_from_payload_error(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings()
            adapter = TwitterAdapter()

        blocked = XuiInvocationResult(
            return_code=0,
            stdout="",
            stderr="",
            payload={
                "outcomes": [
                    {
                        "source_id": "user:nvidia",
                        "error": "blocked_challenge: login wall required",
                    }
                ]
            },
        )

        is_blocked, reason = adapter._detect_xui_block(blocked)
        assert is_blocked is True
        assert reason is not None

    def test_detect_xui_block_ignores_success_output_ids(self):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings()
            adapter = TwitterAdapter()

        successful = XuiInvocationResult(
            return_code=0,
            stdout='{"items":[{"tweet_id":"189429001111"}]}',
            stderr="",
            payload={
                "items": [{"tweet_id": "189429001111"}],
                "outcomes": [{"ok": True, "error": None}],
            },
        )

        is_blocked, reason = adapter._detect_xui_block(successful)
        assert is_blocked is False
        assert reason is None

    def test_prepare_runtime_config_without_existing_base_file(self, tmp_path):
        with patch("src.ingestion.twitter_adapter.get_settings") as mock_settings:
            mock_settings.return_value = _settings(
                twitter_xui_config_path=str(tmp_path / "xui-config.toml"),
                twitter_xui_scroll_pause_min_ms=1400,
                twitter_xui_scroll_pause_max_ms=1400,
                twitter_xui_max_scroll_rounds=4,
            )
            adapter = TwitterAdapter()

        runtime_path, cleanup_path = adapter._prepare_xui_runtime_config()
        assert runtime_path is not None
        assert cleanup_path is not None
        assert runtime_path.exists()

        content = runtime_path.read_text(encoding="utf-8")
        assert "scroll_delay_ms = 1400" in content
        assert content.count("max_scrolls = 4") >= 2
        assert "navigation_timeout_ms = 90000" in content

        cleanup_path.unlink(missing_ok=True)
