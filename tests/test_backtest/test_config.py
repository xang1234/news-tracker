"""Tests for BacktestConfig defaults and environment overrides."""

import os
from unittest.mock import patch

from src.backtest.config import BacktestConfig


class TestDefaults:
    """Test BacktestConfig default values."""

    def test_price_cache_enabled_default(self) -> None:
        config = BacktestConfig()
        assert config.price_cache_enabled is True

    def test_price_cache_ttl_default(self) -> None:
        config = BacktestConfig()
        assert config.price_cache_ttl_days == 1

    def test_forward_horizons_default(self) -> None:
        config = BacktestConfig()
        assert config.default_forward_horizons == [1, 5, 10, 20]

    def test_yfinance_rate_limit_default(self) -> None:
        config = BacktestConfig()
        assert config.yfinance_rate_limit == 5

    def test_auto_version_disabled_by_default(self) -> None:
        config = BacktestConfig()
        assert config.auto_version_on_clustering is False


class TestEnvOverrides:
    """Test BacktestConfig environment variable overrides."""

    def test_price_cache_disabled(self) -> None:
        with patch.dict(os.environ, {"BACKTEST_PRICE_CACHE_ENABLED": "false"}):
            config = BacktestConfig()
            assert config.price_cache_enabled is False

    def test_custom_ttl(self) -> None:
        with patch.dict(os.environ, {"BACKTEST_PRICE_CACHE_TTL_DAYS": "7"}):
            config = BacktestConfig()
            assert config.price_cache_ttl_days == 7

    def test_custom_rate_limit(self) -> None:
        with patch.dict(os.environ, {"BACKTEST_YFINANCE_RATE_LIMIT": "10"}):
            config = BacktestConfig()
            assert config.yfinance_rate_limit == 10

    def test_auto_version_enabled(self) -> None:
        with patch.dict(os.environ, {"BACKTEST_AUTO_VERSION_ON_CLUSTERING": "true"}):
            config = BacktestConfig()
            assert config.auto_version_on_clustering is True
