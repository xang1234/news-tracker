"""Tests for PriceDataFeed: caching, yfinance mocking, forward returns."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.backtest.config import BacktestConfig
from src.backtest.data_feeds import PriceDataFeed


def _make_ohlcv(ticker: str, d: date, close: float = 100.0) -> dict:
    """Helper to create an OHLCV row."""
    return {
        "ticker": ticker,
        "date": d,
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": 1_000_000,
    }


class TestGetOhlcv:
    """Test PriceDataFeed.get_ohlcv()."""

    @pytest.mark.asyncio
    async def test_returns_cached_data(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Returns data from cache when available."""
        row = _make_ohlcv("NVDA", date(2025, 6, 15))
        mock_database.fetch.return_value = [row]

        feed = PriceDataFeed(mock_database)
        result = await feed.get_ohlcv("NVDA", date(2025, 6, 15), date(2025, 6, 15))

        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["close"] == 100.0

    @pytest.mark.asyncio
    async def test_cache_query_sql(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Cache query uses correct SQL."""
        mock_database.fetch.return_value = []
        config = BacktestConfig(price_cache_enabled=False)  # Disable fetch
        feed = PriceDataFeed(mock_database, config=config)

        await feed.get_ohlcv("NVDA", date(2025, 6, 1), date(2025, 6, 30))

        sql = mock_database.fetch.call_args[0][0]
        assert "price_cache" in sql
        assert "ticker = $1" in sql
        assert "date >= $2" in sql
        assert "date <= $3" in sql

    @pytest.mark.asyncio
    async def test_fetches_missing_dates(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Fetches from yfinance when cache misses exist."""
        # Cache returns nothing
        mock_database.fetch.return_value = []

        feed = PriceDataFeed(mock_database)
        fetched_rows = [_make_ohlcv("NVDA", date(2025, 6, 16))]

        with patch.object(
            PriceDataFeed, "_yfinance_download", return_value=fetched_rows,
        ):
            result = await feed.get_ohlcv(
                "NVDA", date(2025, 6, 16), date(2025, 6, 16),
            )

        assert len(result) == 1
        # Verify cache write
        assert mock_database.execute.call_count >= 1
        cache_sql = mock_database.execute.call_args[0][0]
        assert "INSERT INTO price_cache" in cache_sql

    @pytest.mark.asyncio
    async def test_results_sorted_by_date(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Results are sorted by date ascending."""
        rows = [
            _make_ohlcv("NVDA", date(2025, 6, 17)),
            _make_ohlcv("NVDA", date(2025, 6, 15)),
            _make_ohlcv("NVDA", date(2025, 6, 16)),
        ]
        mock_database.fetch.return_value = rows
        config = BacktestConfig(price_cache_enabled=False)
        feed = PriceDataFeed(mock_database, config=config)

        result = await feed.get_ohlcv("NVDA", date(2025, 6, 15), date(2025, 6, 17))

        dates = [r["date"] for r in result]
        assert dates == sorted(dates)

    @pytest.mark.asyncio
    async def test_uppercases_ticker(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Ticker is normalized to uppercase."""
        mock_database.fetch.return_value = []
        config = BacktestConfig(price_cache_enabled=False)
        feed = PriceDataFeed(mock_database, config=config)

        await feed.get_ohlcv("nvda", date(2025, 6, 15), date(2025, 6, 15))

        args = mock_database.fetch.call_args[0]
        assert args[1] == "NVDA"


class TestGetForwardReturns:
    """Test PriceDataFeed.get_forward_returns()."""

    @pytest.mark.asyncio
    async def test_computes_returns(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Computes percentage returns correctly."""
        # Day 0: close=100, Day 1: close=105, Day 5: close=110
        rows = []
        for i in range(10):
            d = date(2025, 6, 15) + timedelta(days=i)
            close = 100.0 + (i * 5.0)
            rows.append(_make_ohlcv("NVDA", d, close))

        mock_database.fetch.return_value = rows
        config = BacktestConfig(price_cache_enabled=False)
        feed = PriceDataFeed(mock_database, config=config)

        result = await feed.get_forward_returns(
            tickers=["NVDA"],
            as_of=date(2025, 6, 15),
            horizons=[1, 5],
        )

        assert "NVDA" in result
        # Day 1: (105 - 100) / 100 = 0.05
        assert result["NVDA"][1] == pytest.approx(0.05)
        # Day 5: (125 - 100) / 100 = 0.25
        assert result["NVDA"][5] == pytest.approx(0.25)

    @pytest.mark.asyncio
    async def test_missing_data_returns_none(
        self,
        mock_database: AsyncMock,
    ) -> None:
        """Returns None when not enough trading days exist."""
        mock_database.fetch.return_value = []
        config = BacktestConfig(price_cache_enabled=False)
        feed = PriceDataFeed(mock_database, config=config)

        result = await feed.get_forward_returns(
            tickers=["XYZ"],
            as_of=date(2025, 6, 15),
            horizons=[1, 5],
        )

        assert result["XYZ"][1] is None
        assert result["XYZ"][5] is None

    @pytest.mark.asyncio
    async def test_uses_default_horizons(
        self,
        mock_database: AsyncMock,
        backtest_config: BacktestConfig,
    ) -> None:
        """Uses config default horizons when not specified."""
        mock_database.fetch.return_value = []
        config = BacktestConfig(price_cache_enabled=False)
        feed = PriceDataFeed(mock_database, config=config)

        result = await feed.get_forward_returns(
            tickers=["NVDA"],
            as_of=date(2025, 6, 15),
        )

        assert set(result["NVDA"].keys()) == {1, 5, 10, 20}


class TestYfinanceDownload:
    """Test the static yfinance download method."""

    def test_calls_yfinance(self) -> None:
        """Downloads data via yfinance."""
        import types

        import pandas as pd

        mock_df = pd.DataFrame({
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Volume": [1000000],
        }, index=[pd.Timestamp("2025-06-15")])

        mock_yf = types.ModuleType("yfinance")
        mock_yf.download = lambda *args, **kwargs: mock_df

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = PriceDataFeed._yfinance_download(
                "NVDA", date(2025, 6, 15), date(2025, 6, 15),
            )

        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"
        assert result[0]["close"] == 103.0

    def test_empty_response(self) -> None:
        """Returns empty list when yfinance returns no data."""
        import types

        import pandas as pd

        mock_yf = types.ModuleType("yfinance")
        mock_yf.download = lambda *args, **kwargs: pd.DataFrame()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            result = PriceDataFeed._yfinance_download(
                "INVALID", date(2025, 6, 15), date(2025, 6, 15),
            )

        assert result == []
