"""Backtest data infrastructure for point-in-time analysis.

Components:
- BacktestConfig: Settings for price caching, forward horizons, rate limits
- PointInTimeService: Temporal queries for themes, documents, metrics
- ModelVersion / ModelVersionRepository: Config snapshot tracking
- PriceDataFeed: OHLCV cache with yfinance fallback
- BacktestRun / BacktestRunRepository: Audit log for backtest executions
"""

from src.backtest.audit import BacktestRun, BacktestRunRepository
from src.backtest.config import BacktestConfig
from src.backtest.data_feeds import PriceDataFeed
from src.backtest.model_versions import ModelVersion, ModelVersionRepository
from src.backtest.point_in_time import PointInTimeService

__all__ = [
    "BacktestConfig",
    "BacktestRun",
    "BacktestRunRepository",
    "ModelVersion",
    "ModelVersionRepository",
    "PointInTimeService",
    "PriceDataFeed",
]
