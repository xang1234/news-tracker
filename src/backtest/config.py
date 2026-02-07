"""Backtest service configuration.

Controls price caching, forward return horizons, yfinance rate limits,
and auto-versioning behavior. All settings can be overridden via
``BACKTEST_*`` environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BacktestConfig(BaseSettings):
    """Configuration for the backtest data infrastructure."""

    model_config = SettingsConfigDict(
        env_prefix="BACKTEST_",
        case_sensitive=False,
        extra="ignore",
    )

    price_cache_enabled: bool = Field(
        default=True,
        description="Enable local DB caching for OHLCV price data",
    )
    price_cache_ttl_days: int = Field(
        default=1,
        ge=1,
        le=365,
        description="Days before cached price rows are considered stale",
    )
    default_forward_horizons: list[int] = Field(
        default=[1, 5, 10, 20],
        description="Default forward return horizons in trading days",
    )
    yfinance_rate_limit: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Max yfinance downloads per minute",
    )
    auto_version_on_clustering: bool = Field(
        default=False,
        description="Automatically snapshot model version when daily clustering runs",
    )
