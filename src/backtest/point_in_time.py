"""Point-in-time data retrieval for backtesting.

Stateless service wrapping ThemeRepository and Database to provide
temporal queries that respect ingestion time (fetched_at) rather than
publication time (timestamp), preventing look-ahead bias.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Protocol

from src.filing.sec_delta_models import SECFilingDeltaEvent
from src.filing.sec_delta_repository import SECFilingDeltaRepository
from src.filing.sec_ownership_events import SECOwnershipEvent, SECOwnershipEventRepository
from src.market_structure import MarketStructureEvent, MarketStructureEventRepository
from src.storage.database import Database
from src.themes.repository import ThemeRepository
from src.themes.schemas import Theme, ThemeMetrics

logger = logging.getLogger(__name__)


class SecDeltaEventReader(Protocol):
    async def list_events_as_of(
        self,
        cik: str,
        *,
        as_of: datetime,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SECFilingDeltaEvent]:
        """Return SEC delta events available at a point in time."""
        ...


class SecOwnershipEventReader(Protocol):
    async def list_events(
        self,
        *,
        issuer_cik: str | None = None,
        filer_cik: str | None = None,
        as_of: datetime | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SECOwnershipEvent]:
        """Return SEC ownership events available at a point in time."""
        ...


class MarketStructureEventReader(Protocol):
    async def list_events(
        self,
        *,
        symbol: str | None = None,
        cusip: str | None = None,
        as_of: datetime | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[MarketStructureEvent]:
        """Return market-structure events available at a point in time."""
        ...


class PointInTimeService:
    """Retrieves data as it existed at any historical point.

    All queries filter on ingestion time (fetched_at) or creation time,
    and respect soft-deleted themes via the deleted_at column.
    """

    def __init__(
        self,
        database: Database,
        theme_repo: ThemeRepository,
        sec_delta_repo: SecDeltaEventReader | None = None,
        sec_ownership_repo: SecOwnershipEventReader | None = None,
        market_structure_repo: MarketStructureEventReader | None = None,
    ) -> None:
        self._db = database
        self._theme_repo = theme_repo
        self._sec_delta_repo = sec_delta_repo or SECFilingDeltaRepository(database)
        self._sec_ownership_repo = sec_ownership_repo or SECOwnershipEventRepository(database)
        self._market_structure_repo = market_structure_repo or MarketStructureEventRepository(
            database
        )

    async def get_themes_as_of(
        self,
        as_of: datetime,
        lifecycle_stages: list[str] | None = None,
        limit: int = 100,
    ) -> list[Theme]:
        """Get themes that were active at a specific point in time.

        A theme is "active at time T" if:
        - created_at <= T, AND
        - (deleted_at IS NULL OR deleted_at > T)

        Args:
            as_of: Point in time to query.
            lifecycle_stages: Optional lifecycle stage filter.
            limit: Maximum themes to return.

        Returns:
            Themes that existed at the given time, ordered by updated_at DESC.
        """
        return await self._theme_repo.get_all_as_of(
            as_of=as_of,
            lifecycle_stages=lifecycle_stages,
            limit=limit,
        )

    async def get_documents_as_of(
        self,
        as_of: datetime,
        since: datetime | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get documents that had been ingested by a specific point in time.

        Filters on fetched_at (ingestion time), NOT timestamp (publication time),
        to prevent look-ahead bias in backtesting.

        Args:
            as_of: Only include documents fetched on or before this time.
            since: Optional lower bound on fetched_at.
            limit: Maximum documents to return.

        Returns:
            List of lightweight document dicts (id, content, embedding,
            authority_score, sentiment, theme_ids, fetched_at).
        """
        conditions = ["fetched_at <= $1", "embedding IS NOT NULL"]
        params: list[Any] = [as_of]
        idx = 2

        if since:
            conditions.append(f"fetched_at >= ${idx}")
            params.append(since)
            idx += 1

        where = " AND ".join(conditions)
        sql = f"""
            SELECT id, content, embedding, authority_score, sentiment,
                   theme_ids, fetched_at
            FROM documents
            WHERE {where}
            ORDER BY fetched_at DESC
            LIMIT ${idx}
        """
        params.append(limit)

        rows = await self._db.fetch(sql, *params)

        results = []
        for row in rows:
            import json

            sentiment = row.get("sentiment")
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment)

            embedding = row.get("embedding")
            if isinstance(embedding, str):
                embedding = [float(x) for x in embedding.strip("[]").split(",")]
            elif isinstance(embedding, list):
                pass
            else:
                embedding = None

            results.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "embedding": embedding,
                    "authority_score": row.get("authority_score"),
                    "sentiment": sentiment,
                    "theme_ids": list(row.get("theme_ids", [])),
                    "fetched_at": row["fetched_at"],
                }
            )

        return results

    async def get_metrics_as_of(
        self,
        theme_id: str,
        as_of: datetime,
        lookback_days: int = 30,
    ) -> list[ThemeMetrics]:
        """Get theme metrics up to a specific point in time.

        Args:
            theme_id: Theme identifier.
            as_of: Upper bound date (inclusive).
            lookback_days: Number of days to look back.

        Returns:
            ThemeMetrics rows within the lookback window, ordered by date ASC.
        """
        start = (as_of - timedelta(days=lookback_days)).date()
        end = as_of.date()
        return await self._theme_repo.get_metrics_range(
            theme_id=theme_id,
            start=start,
            end=end,
        )

    async def get_theme_centroids_as_of(
        self,
        as_of: datetime,
    ) -> dict[str, list[float]]:
        """Get centroid vectors for all themes active at a point in time.

        Args:
            as_of: Point in time to query.

        Returns:
            Dict mapping theme_id to centroid as list of floats.
        """
        themes = await self.get_themes_as_of(as_of=as_of, limit=10000)
        return {t.theme_id: t.centroid.tolist() for t in themes}

    async def get_sec_delta_events_as_of(
        self,
        cik: str,
        *,
        as_of: datetime,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SECFilingDeltaEvent]:
        """Get SEC filing-delta events available at a specific point in time.

        Filters on ``available_at`` through SECFilingDeltaRepository so
        backtests replay only issuer facts known by the historical clock.
        """
        return await self._sec_delta_repo.list_events_as_of(
            cik,
            as_of=as_of,
            event_type=event_type,
            limit=limit,
        )

    async def get_sec_ownership_events_as_of(
        self,
        *,
        issuer_cik: str | None = None,
        filer_cik: str | None = None,
        as_of: datetime,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[SECOwnershipEvent]:
        """Get SEC ownership events known at a specific point in time."""
        return await self._sec_ownership_repo.list_events(
            issuer_cik=issuer_cik,
            filer_cik=filer_cik,
            as_of=as_of,
            event_type=event_type,
            limit=limit,
        )

    async def get_market_structure_events_as_of(
        self,
        *,
        symbol: str | None = None,
        cusip: str | None = None,
        as_of: datetime,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[MarketStructureEvent]:
        """Get market-structure events known at a specific point in time."""
        return await self._market_structure_repo.list_events(
            symbol=symbol,
            cusip=cusip,
            as_of=as_of,
            event_type=event_type,
            limit=limit,
        )
