"""Backtest infrastructure for point-in-time analysis and simulation.

Components:
- BacktestConfig: Settings for price caching, forward horizons, rate limits
- PointInTimeService: Temporal queries for themes, documents, metrics
- ModelVersion / ModelVersionRepository: Config snapshot tracking
- PriceDataFeed: OHLCV cache with yfinance fallback
- BacktestRun / BacktestRunRepository: Audit log for backtest executions
- BacktestEngine: Simulation engine over historical trading days
- BacktestMetrics / CalibrationBucket: Performance metric computation
"""

from src.backtest.audit import BacktestRun, BacktestRunRepository
from src.backtest.config import BacktestConfig
from src.backtest.data_feeds import PriceDataFeed
from src.backtest.engine import BacktestEngine, BacktestResults, DailyBacktestResult
from src.backtest.metrics import BacktestMetrics, CalibrationBucket
from src.backtest.model_versions import ModelVersion, ModelVersionRepository
from src.backtest.point_in_time import PointInTimeService

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BacktestResults",
    "BacktestRun",
    "BacktestRunRepository",
    "CalibrationBucket",
    "DailyBacktestResult",
    "ModelVersion",
    "ModelVersionRepository",
    "PointInTimeService",
    "PriceDataFeed",
]
