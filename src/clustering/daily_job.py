"""Daily batch clustering job for theme reassignment, new theme detection, and metrics.

Runs as an offline batch process (distinct from the real-time ClusteringWorker):
1. Fetches documents with FinBERT embeddings from a 24h window
2. Batch cosine similarity against all theme centroids (numpy matrix multiply)
3. Three-tier assignment: strong (EMA update), weak (assign only), unassigned
4. Detects emerging themes from unassigned document candidates
5. Computes basic daily metrics (sentiment, authority) per theme
6. Classifies lifecycle stages and detects transitions
7. Weekly theme merge on Mondays to consolidate similar themes

Designed for external cron scheduling: ``0 4 * * * news-tracker daily-clustering``
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from src.clustering.config import ClusteringConfig
from src.clustering.schemas import ThemeCluster
from src.clustering.service import BERTopicService
from src.storage.database import Database
from src.storage.repository import DocumentRepository
from src.themes.lifecycle import LifecycleClassifier
from src.themes.repository import ThemeRepository
from src.themes.schemas import Theme, ThemeMetrics
from src.themes.transitions import LifecycleTransition

logger = logging.getLogger(__name__)


@dataclass
class DailyClusteringResult:
    """Summary of a daily clustering batch run."""

    date: date
    documents_fetched: int = 0
    documents_assigned: int = 0
    documents_unassigned: int = 0
    new_themes_created: int = 0
    themes_merged: int = 0
    metrics_computed: int = 0
    alerts_generated: int = 0
    lifecycle_transitions: list[LifecycleTransition] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


async def run_daily_clustering(
    database: Database,
    target_date: date | None = None,
    config: ClusteringConfig | None = None,
) -> DailyClusteringResult:
    """
    Run the daily batch clustering pipeline.

    Args:
        database: Connected Database instance (caller manages lifecycle).
        target_date: Date to process (default: today UTC).
        config: Clustering configuration (default: from env).

    Returns:
        DailyClusteringResult with counts and any errors.
    """
    config = config or ClusteringConfig()
    target_date = target_date or datetime.now(timezone.utc).date()
    result = DailyClusteringResult(date=target_date)
    start_time = time.monotonic()

    doc_repo = DocumentRepository(database)
    theme_repo = ThemeRepository(database)

    # Phase 1: Time window
    since = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    until = since + timedelta(days=1)

    # Phase 2: Fetch documents
    try:
        docs = await doc_repo.get_with_embeddings_since(since, until)
        result.documents_fetched = len(docs)
    except Exception as e:
        logger.exception("Failed to fetch documents")
        result.errors.append(f"fetch_docs: {e}")
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    if not docs:
        logger.info("No documents with embeddings for %s", target_date)
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    # Phase 3: Load existing themes
    try:
        themes = await theme_repo.get_all(limit=500)
    except Exception as e:
        logger.exception("Failed to load themes")
        result.errors.append(f"load_themes: {e}")
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    if not themes:
        logger.info("No existing themes — all %d docs unassigned", len(docs))
        result.documents_unassigned = len(docs)
        result.elapsed_seconds = time.monotonic() - start_time
        return result

    # Phase 4: Batch cosine similarity
    embeddings = np.array([d["embedding"] for d in docs], dtype=np.float32)
    centroids = np.vstack([t.centroid for t in themes])
    sim_matrix = _batch_cosine_similarity(embeddings, centroids)

    # Phase 5: Three-tier routing
    assign_threshold = config.similarity_threshold_assign
    new_threshold = config.similarity_threshold_new

    strong_assignments: list[tuple[int, int, float]] = []  # (doc_idx, theme_idx, sim)
    weak_assignments: list[tuple[int, int, float]] = []
    unassigned_indices: list[int] = []

    for i in range(len(docs)):
        max_idx = int(np.argmax(sim_matrix[i]))
        max_sim = float(sim_matrix[i, max_idx])

        if max_sim >= assign_threshold:
            strong_assignments.append((i, max_idx, max_sim))
        elif max_sim >= new_threshold:
            weak_assignments.append((i, max_idx, max_sim))
        else:
            unassigned_indices.append(i)

    result.documents_assigned = len(strong_assignments) + len(weak_assignments)
    result.documents_unassigned = len(unassigned_indices)

    # Phase 6: Persist assignments
    all_assignments = strong_assignments + weak_assignments
    for doc_idx, theme_idx, _sim in all_assignments:
        try:
            await doc_repo.update_themes(docs[doc_idx]["id"], [themes[theme_idx].theme_id])
        except Exception as e:
            logger.error("Failed to assign doc %s: %s", docs[doc_idx]["id"], e)
            result.errors.append(f"assign:{docs[doc_idx]['id']}: {e}")

    # Phase 7: EMA centroid update for strong assignments
    lr = config.centroid_learning_rate
    theme_centroid_updates: dict[str, np.ndarray] = {}

    for doc_idx, theme_idx, _sim in strong_assignments:
        theme = themes[theme_idx]
        tid = theme.theme_id
        current = theme_centroid_updates.get(tid, theme.centroid.copy())
        current = (1 - lr) * current + lr * embeddings[doc_idx]
        theme_centroid_updates[tid] = current.astype(np.float32)

    for tid, new_centroid in theme_centroid_updates.items():
        try:
            await theme_repo.update_centroid(tid, new_centroid)
        except Exception as e:
            logger.error("Failed to update centroid for %s: %s", tid, e)
            result.errors.append(f"centroid:{tid}: {e}")

    # Phase 8: Update document_count per theme
    theme_doc_counts: dict[str, int] = {}
    for _doc_idx, theme_idx, _sim in all_assignments:
        tid = themes[theme_idx].theme_id
        theme_doc_counts[tid] = theme_doc_counts.get(tid, 0) + 1

    for tid, count in theme_doc_counts.items():
        try:
            await database.execute(
                "UPDATE themes SET document_count = document_count + $2 WHERE theme_id = $1",
                tid,
                count,
            )
        except Exception as e:
            logger.error("Failed to update document_count for %s: %s", tid, e)
            result.errors.append(f"doc_count:{tid}: {e}")

    # Phase 9: New theme detection
    new_theme_ids: list[str] = []
    if unassigned_indices:
        try:
            new_theme_ids = await _detect_new_themes(
                docs, embeddings, unassigned_indices, themes, config, theme_repo,
            )
            result.new_themes_created = len(new_theme_ids)
        except Exception as e:
            logger.exception("New theme detection failed")
            result.errors.append(f"new_themes: {e}")

    # Phase 10: Compute basic daily metrics
    try:
        metrics_count = await _compute_daily_metrics(
            target_date, themes, theme_doc_counts, doc_repo, theme_repo, since, until,
        )
        result.metrics_computed = metrics_count
    except Exception as e:
        logger.exception("Metrics computation failed")
        result.errors.append(f"metrics: {e}")

    # Phase 11: Lifecycle classification
    try:
        transitions = await _classify_lifecycle_stages(
            target_date, themes, theme_repo,
        )
        result.lifecycle_transitions = transitions
        for t in transitions:
            if t.is_alertable:
                logger.info(
                    "Lifecycle transition: %s %s → %s (%s)",
                    t.theme_id, t.from_stage, t.to_stage, t.alert_message,
                )
    except Exception as e:
        logger.exception("Lifecycle classification failed")
        result.errors.append(f"lifecycle: {e}")

    # Weekly maintenance: theme merge on Mondays
    if target_date.weekday() == 0:  # Monday
        try:
            merge_count = await _run_weekly_merge(themes, config, theme_repo, database)
            result.themes_merged = merge_count
        except Exception as e:
            logger.exception("Weekly merge failed")
            result.errors.append(f"merge: {e}")

    # Phase 12: Alert generation (gated by settings.alerts_enabled)
    try:
        alerts_count = await _generate_daily_alerts(
            target_date, themes, theme_doc_counts, theme_repo,
            result.lifecycle_transitions, new_theme_ids, database,
        )
        result.alerts_generated = alerts_count
    except Exception as e:
        logger.exception("Alert generation failed")
        result.errors.append(f"alerts: {e}")

    result.elapsed_seconds = time.monotonic() - start_time
    logger.info(
        "Daily clustering complete for %s: fetched=%d assigned=%d unassigned=%d "
        "new_themes=%d merged=%d metrics=%d alerts=%d errors=%d elapsed=%.2fs",
        target_date,
        result.documents_fetched,
        result.documents_assigned,
        result.documents_unassigned,
        result.new_themes_created,
        result.themes_merged,
        result.metrics_computed,
        result.alerts_generated,
        len(result.errors),
        result.elapsed_seconds,
    )

    return result


# ── Helper functions ─────────────────────────────────────────


def _batch_cosine_similarity(
    embeddings: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between document embeddings and theme centroids.

    Uses normalized matrix multiply for O(n_docs × n_themes) computation
    without Python loops.

    Args:
        embeddings: (n_docs, dim) document embedding matrix.
        centroids: (n_themes, dim) theme centroid matrix.

    Returns:
        (n_docs, n_themes) similarity matrix with values in [-1, 1].
    """
    # Normalize embeddings
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms = np.where(emb_norms == 0, 1.0, emb_norms)
    emb_normalized = embeddings / emb_norms

    # Normalize centroids
    cen_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cen_norms = np.where(cen_norms == 0, 1.0, cen_norms)
    cen_normalized = centroids / cen_norms

    return emb_normalized @ cen_normalized.T


def _theme_to_cluster(theme: Theme) -> ThemeCluster:
    """
    Convert a DB Theme to an in-memory ThemeCluster for BERTopicService.

    Translates flat top_keywords strings to (word, score) topic_words tuples
    with descending synthetic scores for compatibility with BERTopicService
    methods that operate on topic_words.
    """
    topic_words = [
        (kw, 1.0 - i * 0.01) for i, kw in enumerate(theme.top_keywords)
    ]
    return ThemeCluster(
        theme_id=theme.theme_id,
        name=theme.name,
        topic_words=topic_words,
        centroid=theme.centroid.copy(),
        document_count=theme.document_count,
        document_ids=[],
        created_at=theme.created_at,
        updated_at=theme.updated_at,
        metadata=dict(theme.metadata),
    )


def _cluster_to_theme(cluster: ThemeCluster) -> Theme:
    """
    Convert an in-memory ThemeCluster to a DB Theme for persistence.

    Extracts flat keyword strings from topic_words tuples and sets
    lifecycle_stage from metadata if present.
    """
    lifecycle = cluster.metadata.get("lifecycle_stage", "emerging")
    return Theme(
        theme_id=cluster.theme_id,
        name=cluster.name,
        centroid=cluster.centroid.copy(),
        top_keywords=[w for w, _ in cluster.topic_words],
        document_count=cluster.document_count,
        lifecycle_stage=lifecycle,
        metadata=dict(cluster.metadata),
    )


async def _detect_new_themes(
    docs: list[dict[str, Any]],
    embeddings: np.ndarray,
    unassigned_indices: list[int],
    existing_themes: list[Theme],
    config: ClusteringConfig,
    theme_repo: ThemeRepository,
) -> list[str]:
    """
    Detect new themes from unassigned documents via BERTopicService.check_new_themes().

    Populates a BERTopicService instance with existing theme data, then feeds
    unassigned candidates through its mini-clustering pipeline.

    Returns:
        List of created theme IDs (empty if none detected).
    """
    min_candidates = max(3, config.hdbscan_min_cluster_size // 2)
    if len(unassigned_indices) < min_candidates:
        logger.info(
            "Too few unassigned docs (%d) for new theme detection (need %d)",
            len(unassigned_indices),
            min_candidates,
        )
        return []

    # Build BERTopicService pre-populated with existing themes
    service = BERTopicService(config=config)
    for theme in existing_themes:
        service._themes[theme.theme_id] = _theme_to_cluster(theme)
    service._initialized = True

    # Prepare candidates: (doc_id, text, embedding) triples
    candidates = [
        (docs[i]["id"], docs[i]["content"], embeddings[i])
        for i in unassigned_indices
    ]

    new_clusters = service.check_new_themes(candidates)

    # Persist new themes to DB
    created_ids: list[str] = []
    for cluster in new_clusters:
        theme = _cluster_to_theme(cluster)
        try:
            await theme_repo.create(theme)
            created_ids.append(theme.theme_id)
        except Exception as e:
            logger.error("Failed to persist new theme %s: %s", cluster.theme_id, e)

    return created_ids


async def _compute_daily_metrics(
    target_date: date,
    themes: list[Theme],
    theme_doc_counts: dict[str, int],
    doc_repo: DocumentRepository,
    theme_repo: ThemeRepository,
    since: datetime,
    until: datetime,
) -> int:
    """
    Compute and persist basic daily metrics for each active theme.

    Active = any theme that had documents assigned today or already existed.
    Metrics: document_count, sentiment_score, avg_authority, bullish_ratio.
    Leaves volume_zscore, velocity, acceleration as None (Feature 3.1 scope).

    Returns:
        Number of metrics records persisted.
    """
    computed = 0

    for theme in themes:
        tid = theme.theme_id
        day_count = theme_doc_counts.get(tid, 0)

        # Skip themes with no activity today
        if day_count == 0:
            continue

        try:
            sentiments = await doc_repo.get_sentiments_for_theme(tid, since, until)
            metrics = _aggregate_sentiment_metrics(tid, target_date, day_count, sentiments)
            await theme_repo.add_metrics(metrics)
            computed += 1
        except Exception as e:
            logger.error("Failed to compute metrics for theme %s: %s", tid, e)

    return computed


def _aggregate_sentiment_metrics(
    theme_id: str,
    target_date: date,
    document_count: int,
    sentiments: list[dict[str, Any]],
) -> ThemeMetrics:
    """
    Aggregate sentiment records into a ThemeMetrics row.

    Computes:
    - sentiment_score: mean of (positive - negative) scores weighted equally
    - avg_authority: mean authority_score (excluding None)
    - bullish_ratio: fraction of documents with positive sentiment label
    """
    sentiment_score = None
    avg_authority = None
    bullish_ratio = None

    if sentiments:
        # Sentiment score: mean of (positive - negative) from individual scores
        pos_neg_scores = []
        positive_count = 0
        authority_values = []

        for s in sentiments:
            sent = s.get("sentiment")
            if sent and isinstance(sent, dict):
                scores = sent.get("scores", {})
                pos = scores.get("positive", 0.0)
                neg = scores.get("negative", 0.0)
                pos_neg_scores.append(pos - neg)

                label = sent.get("label", "")
                if label == "positive":
                    positive_count += 1

            auth = s.get("authority_score")
            if auth is not None:
                authority_values.append(auth)

        if pos_neg_scores:
            sentiment_score = sum(pos_neg_scores) / len(pos_neg_scores)
            bullish_ratio = positive_count / len(pos_neg_scores)

        if authority_values:
            avg_authority = sum(authority_values) / len(authority_values)

    return ThemeMetrics(
        theme_id=theme_id,
        date=target_date,
        document_count=document_count,
        sentiment_score=sentiment_score,
        avg_authority=avg_authority,
        bullish_ratio=bullish_ratio,
        # volume_zscore, velocity, acceleration left None (Feature 3.1)
    )


async def _run_weekly_merge(
    themes: list[Theme],
    config: ClusteringConfig,
    theme_repo: ThemeRepository,
    database: Database,
) -> int:
    """
    Run weekly theme merge via BERTopicService.merge_similar_themes().

    For each absorbed theme:
    - Delete from DB
    - Update survivor's centroid, document_count, top_keywords

    Returns:
        Number of themes merged (absorbed).
    """
    if len(themes) < 2:
        return 0

    service = BERTopicService(config=config)
    for theme in themes:
        service._themes[theme.theme_id] = _theme_to_cluster(theme)
    service._initialized = True

    merge_results = service.merge_similar_themes()

    if not merge_results:
        return 0

    for absorbed_id, survivor_id in merge_results:
        try:
            # Delete absorbed theme from DB
            await theme_repo.delete(absorbed_id)

            # Get the updated survivor from the service's internal state
            # Note: merge re-keys themes, so look up by iterating
            survivor_cluster = None
            for cluster in service._themes.values():
                merged_from = cluster.metadata.get("merged_from", [])
                if absorbed_id in merged_from:
                    survivor_cluster = cluster
                    break

            if survivor_cluster:
                await theme_repo.update_centroid(
                    survivor_cluster.theme_id, survivor_cluster.centroid,
                )
                await theme_repo.update(survivor_cluster.theme_id, {
                    "document_count": survivor_cluster.document_count,
                    "top_keywords": [w for w, _ in survivor_cluster.topic_words],
                })

        except Exception as e:
            logger.error(
                "Failed to finalize merge %s → %s: %s",
                absorbed_id, survivor_id, e,
            )

    logger.info("Weekly merge: %d themes absorbed", len(merge_results))
    return len(merge_results)


async def _classify_lifecycle_stages(
    target_date: date,
    themes: list[Theme],
    theme_repo: ThemeRepository,
) -> list[LifecycleTransition]:
    """
    Classify lifecycle stages for all themes and persist stage updates.

    Fetches a 7-day metrics window per theme, classifies the stage, detects
    transitions, and updates the DB lifecycle_stage field.

    Returns:
        List of detected LifecycleTransition records (only actual changes).
    """
    classifier = LifecycleClassifier()
    transitions: list[LifecycleTransition] = []
    lookback = timedelta(days=7)
    start = target_date - lookback
    end = target_date

    for theme in themes:
        try:
            metrics_history = await theme_repo.get_metrics_range(
                theme.theme_id, start, end,
            )
            new_stage, confidence = classifier.classify(theme, metrics_history)
            transition = classifier.detect_transition(theme, new_stage, confidence)

            if transition:
                transitions.append(transition)
                await theme_repo.update(theme.theme_id, {
                    "lifecycle_stage": new_stage,
                })

        except Exception as e:
            logger.error(
                "Failed to classify lifecycle for %s: %s", theme.theme_id, e,
            )

    return transitions


async def _generate_daily_alerts(
    target_date: date,
    themes: list[Theme],
    theme_doc_counts: dict[str, int],
    theme_repo: ThemeRepository,
    lifecycle_transitions: list[LifecycleTransition],
    new_theme_ids: list[str],
    database: Database,
) -> int:
    """
    Generate alerts based on daily metrics analysis (Phase 12).

    Gated by ``settings.alerts_enabled``. Fetches today's and yesterday's
    metrics, runs all trigger functions, and persists filtered alerts.

    Returns:
        Number of alerts persisted.
    """
    from src.config.settings import get_settings

    settings = get_settings()
    if not settings.alerts_enabled:
        return 0

    from src.alerts.config import AlertConfig
    from src.alerts.repository import AlertRepository
    from src.alerts.service import AlertService

    # Only generate alerts for themes that had activity today
    active_theme_ids = {tid for tid, count in theme_doc_counts.items() if count > 0}
    if not active_theme_ids and not new_theme_ids and not lifecycle_transitions:
        return 0

    # Fetch today's and yesterday's metrics for active themes
    yesterday = target_date - timedelta(days=1)
    today_metrics_map: dict[str, ThemeMetrics] = {}
    yesterday_metrics_map: dict[str, ThemeMetrics] = {}

    for theme in themes:
        if theme.theme_id not in active_theme_ids:
            continue
        try:
            today_rows = await theme_repo.get_metrics_range(
                theme.theme_id, target_date, target_date,
            )
            if today_rows:
                today_metrics_map[theme.theme_id] = today_rows[0]

            yesterday_rows = await theme_repo.get_metrics_range(
                theme.theme_id, yesterday, yesterday,
            )
            if yesterday_rows:
                yesterday_metrics_map[theme.theme_id] = yesterday_rows[0]
        except Exception as e:
            logger.error("Failed to fetch metrics for alert gen %s: %s", theme.theme_id, e)

    # Build AlertService with optional Redis
    alert_config = AlertConfig()
    alert_repo = AlertRepository(database)

    redis_client = None
    try:
        import redis.asyncio as redis_lib
        redis_client = redis_lib.from_url(
            str(settings.redis_url), encoding="utf-8", decode_responses=True,
        )
    except Exception as e:
        logger.warning("Redis unavailable for alert dedup: %s", e)

    service = AlertService(
        config=alert_config,
        alert_repo=alert_repo,
        redis_client=redis_client,
    )

    try:
        alerts = await service.generate_alerts(
            themes=themes,
            today_metrics_map=today_metrics_map,
            yesterday_metrics_map=yesterday_metrics_map,
            lifecycle_transitions=lifecycle_transitions,
            new_theme_ids=new_theme_ids,
        )
        return len(alerts)
    finally:
        if redis_client is not None:
            try:
                await redis_client.aclose()
            except Exception:
                pass
