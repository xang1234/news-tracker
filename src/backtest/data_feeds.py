"""Price data feed with caching for backtest forward returns.

Provides OHLCV data via a cache-first strategy: check the price_cache
table first, then fetch missing data from yfinance. Rate limiting is
applied at the I/O boundary (before yfinance download) via asyncio.Semaphore.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.backtest.config import BacktestConfig
from src.storage.database import Database

logger = logging.getLogger(__name__)


class PriceDataFeed:
    """OHLCV price data with DB caching and yfinance fallback."""

    def __init__(
        self,
        database: Database,
        config: BacktestConfig | None = None,
    ) -> None:
        self._db = database
        self._config = config or BacktestConfig()
        self._semaphore = asyncio.Semaphore(1)

    async def get_ohlcv(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Get OHLCV data for a ticker in a date range.

        Cache-first: returns cached rows where available, fetches
        missing dates from yfinance.

        Args:
            ticker: Ticker symbol (e.g., "NVDA").
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            List of dicts with keys: ticker, date, open, high, low,
            close, volume. Ordered by date ascending.
        """
        ticker = ticker.upper()

        # Fetch cached rows
        cached = await self._get_cached(ticker, start, end)
        cached_dates = {row["date"] for row in cached}

        # Determine missing dates
        all_dates = set()
        current = start
        while current <= end:
            all_dates.add(current)
            current += timedelta(days=1)

        missing_dates = all_dates - cached_dates

        if missing_dates and self._config.price_cache_enabled:
            # Fetch from yfinance and cache
            min_missing = min(missing_dates)
            max_missing = max(missing_dates)
            fetched = await self._fetch_and_cache(ticker, min_missing, max_missing)
            cached.extend(fetched)

        # Filter to requested range and sort
        result = [
            row for row in cached
            if start <= row["date"] <= end
        ]
        result.sort(key=lambda r: r["date"])
        return result

    async def get_forward_returns(
        self,
        tickers: list[str],
        as_of: date,
        horizons: list[int] | None = None,
    ) -> dict[str, dict[int, float | None]]:
        """Compute forward returns for tickers from a given date.

        Args:
            tickers: List of ticker symbols.
            as_of: Starting date for return calculation.
            horizons: Forward horizons in trading days (default from config).

        Returns:
            Nested dict: {ticker: {horizon: return_pct or None}}.
            Returns are expressed as fractions (0.05 = 5%).
        """
        horizons = horizons or self._config.default_forward_horizons
        max_horizon = max(horizons) if horizons else 20

        # Fetch enough data to cover the max horizon (add buffer for weekends)
        end = as_of + timedelta(days=max_horizon * 2)

        results: dict[str, dict[int, float | None]] = {}
        for ticker in tickers:
            data = await self.get_ohlcv(ticker, as_of, end)
            trading_days = [row for row in data if row["date"] >= as_of]

            if not trading_days:
                results[ticker] = {h: None for h in horizons}
                continue

            base_price = trading_days[0]["close"]
            returns: dict[int, float | None] = {}
            for h in horizons:
                if h < len(trading_days) and base_price > 0:
                    future_price = trading_days[h]["close"]
                    returns[h] = (future_price - base_price) / base_price
                else:
                    returns[h] = None
            results[ticker] = returns

        return results

    async def _get_cached(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Fetch cached OHLCV rows from the database."""
        sql = """
            SELECT ticker, date, open, high, low, close, volume
            FROM price_cache
            WHERE ticker = $1 AND date >= $2 AND date <= $3
            ORDER BY date ASC
        """
        rows = await self._db.fetch(sql, ticker, start, end)
        return [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
            for row in rows
        ]

    async def _fetch_and_cache(
        self,
        ticker: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Fetch from yfinance and store in cache.

        Rate limiting is applied via semaphore before the I/O call.
        """
        async with self._semaphore:
            rows = await asyncio.to_thread(
                self._yfinance_download, ticker, start, end,
            )

        # Store in cache
        for row in rows:
            sql = """
                INSERT INTO price_cache (ticker, date, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    fetched_at = NOW()
            """
            await self._db.execute(
                sql,
                row["ticker"],
                row["date"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["volume"],
            )

        return rows

    @staticmethod
    def _yfinance_download(
        ticker: str,
        start: date,
        end: date,
    ) -> list[dict[str, Any]]:
        """Download OHLCV data from yfinance (blocking).

        Isolated as a static method for easy mocking in tests.

        Args:
            ticker: Ticker symbol.
            start: Start date.
            end: End date (inclusive, but yfinance uses exclusive end).

        Returns:
            List of OHLCV dicts.
        """
        import yfinance as yf

        # yfinance end is exclusive, add one day
        end_exclusive = end + timedelta(days=1)
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=end_exclusive.isoformat(),
            progress=False,
        )

        if df.empty:
            return []

        results = []
        for idx, row in df.iterrows():
            results.append({
                "ticker": ticker,
                "date": idx.date() if hasattr(idx, "date") else idx,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })

        return results
